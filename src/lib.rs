//use ndarray::prelude::*;
#![feature(associated_type_bounds)]
use numpy::ndarray::prelude::*;
use std::usize;
use tokenizers::tokenizer;

use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

mod python;
mod test;

use python::ToPyObjectConsume;

pub trait Node: Send {
    type Output: ToPyObjectConsume;
    fn get(&self, index: usize) -> Option<Self::Output>;
    fn len(&self) -> Option<usize>;
    fn next(&mut self) -> Option<Self::Output>;
}

pub struct NoLabel();
pub struct Span(Option<(usize, usize)>);

//pub trait Label: ToPyObjectConsume {
pub trait Label {
    type Tokenized: TokenizedLabel;
    fn tokenize(self, encoding: &tokenizer::Encoding, starting_index: usize) -> Self::Tokenized;
}

impl Label for Span {
    type Tokenized = TokenizedSpan;
    fn tokenize(self, encoding: &tokenizer::Encoding, starting_index: usize) -> TokenizedSpan {
        if let Some((mut start, mut end)) = self.0 {
            if start < starting_index {
                return TokenizedSpan(None);
            }
            start -= starting_index;
            end -= starting_index;
            if let (Some(start), Some(end)) = (
                encoding.char_to_token(start, 1),
                encoding.char_to_token(end, 1),
            ) {
                return TokenizedSpan(Some((start, end)));
            }
        }
        TokenizedSpan(None)
    }
}

impl Label for NoLabel {
    type Tokenized = NoTokenizedLabel;
    fn tokenize(self, encoding: &tokenizer::Encoding, starting_index: usize) -> NoTokenizedLabel {
        NoTokenizedLabel
    }
}

pub struct NoTokenizedLabel;
pub struct TokenizedSpan(Option<(usize, usize)>);

pub trait TokenizedLabel: Sized {
    type Batch: BatchLabel;
    fn to_batch(selfs: Vec<Self>) -> Self::Batch;
}
impl TokenizedLabel for NoTokenizedLabel {
    type Batch = NoBatchLabel;
    fn to_batch(_selfs: Vec<Self>) -> NoBatchLabel {
        NoBatchLabel
    }
}

impl TokenizedLabel for TokenizedSpan {
    type Batch = BatchSpan;
    fn to_batch(selfs: Vec<Self>) -> Self::Batch {
        BatchSpan
    }
}

pub struct NoBatchLabel;
pub struct BatchSpan;

impl BatchLabel for NoBatchLabel {}
impl BatchLabel for BatchSpan {}

pub struct Text<T: Label> {
    text: String,
    label: T,
}

#[derive(Debug)]
pub struct Encoding {
    input_ids: Array1<u32>,
    pad_token: u32,
}

pub struct TokenizedText<T: TokenizedLabel> {
    encoding: Encoding,
    label: T,
}

impl Encoding {
    pub fn from_tokenizer_encoding(encoding: tokenizer::Encoding, pad_token: u32) -> Encoding {
        //let tokenizer::Encoding { ids: input_ids, .. } = encoding; // Sadly private so have to
        //clone :(
        let input_ids = encoding.get_ids().to_vec();
        let input_ids = Array::from_vec(input_ids);
        Encoding {
            input_ids: input_ids,
            pad_token,
        }
    }
}

pub struct BatchAnswer {}

pub trait BatchLabel {}

pub struct BatchEncoding {
    input_ids: Array2<u32>,
    pad_token: u32,
}
pub struct Batch<T: BatchLabel> {
    encoding: BatchEncoding,
    labels: T,
}

pub struct Tokenizer<S: Label, T: Node<Output = Text<S>>> {
    loader: T,
    tokenizer: tokenizer::Tokenizer,
}

impl<S: Label, T: Node<Output = Text<S>>> Tokenizer<S, T> {
    fn new<R: AsRef<str>>(loader: T, tokenizer: R) -> Result<Tokenizer<S, T>, tokenizer::Error> {
        let tokenizer = tokenizer::Tokenizer::from_pretrained(tokenizer, None)?;
        Ok(Tokenizer { loader, tokenizer })
    }
    fn tokenize(&self, input: Text<S>) -> TokenizedText<S::Tokenized> {
        let tokens = self
            .tokenizer
            .encode(input.text, false)
            .expect("Failed to tokenize");
        let label = input.label.tokenize(&tokens, 0);
        TokenizedText {
            encoding: Encoding::from_tokenizer_encoding(
                tokens,
                self.tokenizer.get_padding().map_or(0, |pad| pad.pad_id),
            ),
            label: label,
        }
    }
}

impl<S: Label, T: Node<Output = Text<S>>> Node for Tokenizer<S, T> {
    type Output = TokenizedText<S::Tokenized>;
    fn get(&self, index: usize) -> Option<TokenizedText<S::Tokenized>> {
        self.loader.get(index).map(|sample| self.tokenize(sample))
    }
    fn len(&self) -> Option<usize> {
        self.loader.len()
    }
    fn next(&mut self) -> Option<TokenizedText<S::Tokenized>> {
        self.loader.next().map(|sample| self.tokenize(sample))
    }
}

pub struct TxtLoader {
    lines: io::Lines<io::BufReader<File>>,
}

impl TxtLoader {
    fn new<P: AsRef<Path>>(file: P) -> io::Result<TxtLoader> {
        let file = File::open(file)?;
        Ok(TxtLoader {
            lines: io::BufReader::new(file).lines(),
        })
    }
}

impl Node for TxtLoader {
    type Output = Text<NoLabel>;
    // Not implemented for performance reasons
    fn get(&self, _index: usize) -> Option<Self::Output> {
        None
    }
    fn len(&self) -> Option<usize> {
        None
    }
    fn next(&mut self) -> Option<Self::Output> {
        self.lines.next().map(|line| Text {
            text: line.expect("Failed to read line"),
            label: NoLabel(),
        })
    }
}

pub struct StaticBatcher<S: TokenizedLabel, T: Node<Output = TokenizedText<S>>> {
    tokenizer: T,
    batch_size: usize,
    seq_length: usize,
}

impl<S: TokenizedLabel, T: Node<Output = TokenizedText<S>>> StaticBatcher<S, T> {
    pub fn new(
        tokenizer: T,
        batch_size: usize,
        seq_length: usize,
    ) -> Result<StaticBatcher<S, T>, String> {
        Ok(StaticBatcher {
            tokenizer,
            batch_size,
            seq_length,
        })
    }
    pub fn create_batch(&self, samples: Vec<TokenizedText<S>>) -> Batch<S::Batch> {
        let mut inputs: Vec<Array2<u32>> = Vec::new();
        let mut labels: Vec<S> = Vec::new();
        let mut pad_token = 0;
        let len = samples.len();
        for (i, sample) in samples.into_iter().enumerate() {
            let TokenizedText { encoding, label } = sample;
            labels.push(label);
            let Encoding {
                input_ids,
                pad_token: current_pad_token,
            } = encoding;
            let arrays = vec![input_ids];
            pad_token = current_pad_token;
            for (j, array) in arrays.iter().enumerate() {
                match inputs.get_mut(j) {
                    Some(matrix) => {
                        let mut len = array.len();
                        if len > self.seq_length {
                            len = self.seq_length;
                        }
                        matrix
                            .slice_mut(s![i, 0..len])
                            .assign(&array.slice(s![..len]));
                    }
                    None => {
                        let mut matrix = Array2::zeros((len, self.seq_length));
                        matrix.fill(pad_token);
                        let mut len = array.len();
                        if len > self.seq_length {
                            len = self.seq_length;
                        }
                        matrix
                            .slice_mut(s![i, 0..len])
                            .assign(&array.slice(s![..len]));
                        inputs.push(matrix);
                    }
                }
            }
        }
        let input_ids = inputs.pop().unwrap();
        Batch {
            encoding: BatchEncoding {
                input_ids,
                pad_token,
            },
            labels: S::to_batch(labels),
        }
    }
}

impl<S: TokenizedLabel, T: Node<Output = TokenizedText<S>>> Node for StaticBatcher<S, T> {
    type Output = Batch<S::Batch>;
    fn next(&mut self) -> Option<Batch<S::Batch>> {
        let mut samples: Vec<TokenizedText<S>> = Vec::new();
        for _ in 0..self.batch_size {
            match self.tokenizer.next() {
                Some(sample) => samples.push(sample),
                None => break,
            }
        }
        if samples.is_empty() {
            None
        } else {
            Some(self.create_batch(samples))
        }
    }
    fn get(&self, index: usize) -> Option<Batch<S::Batch>> {
        let index = index * self.batch_size;
        let mut samples: Vec<TokenizedText<S>> = Vec::new();
        for i in index..index + self.batch_size {
            match self.tokenizer.get(i) {
                Some(sample) => samples.push(sample),
                None => break,
            }
        }
        if samples.is_empty() {
            None
        } else {
            Some(self.create_batch(samples))
        }
    }
    fn len(&self) -> Option<usize> {
        match self.tokenizer.len() {
            Some(len) => Some(len / self.batch_size),
            None => None,
        }
    }
}

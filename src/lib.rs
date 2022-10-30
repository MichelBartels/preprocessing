//use ndarray::prelude::*;
#![feature(associated_type_bounds)]
use numpy::ndarray::prelude::*;
use std::usize;
use tokenizers::tokenizer;

mod datasets;
mod python;
mod test;

use python::ToPyObjectConsume;

pub trait Node: Send {
    type Output: ToPyObjectConsume;
    fn get(&self, index: usize) -> Option<Self::Output>;
    fn len(&self) -> Option<usize>;
    fn next(&mut self) -> Option<Self::Output>;
}

#[derive(Clone)]
pub struct NoLabel();
#[derive(Clone)]
pub struct Span(Option<(usize, usize)>);

//pub trait Label: ToPyObjectConsume {
pub trait Label: Clone + ToPyObjectConsume {
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
            let sequence_index = encoding.n_sequences() - 1;
            if let (Some(start), Some(end)) = (
                encoding.char_to_token(start, sequence_index),
                encoding.char_to_token(end, sequence_index),
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

pub trait TokenizedLabel: Sized + ToPyObjectConsume {
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
        let mut start = Vec::new();
        let mut end = Vec::new();
        for span in selfs.into_iter() {
            match span.0 {
                Some((start_index, end_index)) => {
                    start.push(start_index);
                    end.push(end_index);
                }
                None => {
                    start.push(0);
                    end.push(0);
                }
            };
        }
        let start = Array1::from_vec(start);
        let end = Array1::from_vec(end);
        BatchSpan { start, end }
    }
}

pub struct NoBatchLabel;
pub struct BatchSpan {
    start: Array1<usize>,
    end: Array1<usize>,
}

impl BatchLabel for NoBatchLabel {}
impl BatchLabel for BatchSpan {}

#[derive(Clone)]
pub struct Text<T: Label> {
    text: String,
    label: T,
}

#[derive(Clone)]
pub struct TextPair<T: Label> {
    text: (String, String),
    label: T,
}

pub trait Sample {
    type Label: Label;
    fn tokenize(
        self,
        tokenizer: &tokenizer::Tokenizer,
    ) -> TokenizedText<<<Self as Sample>::Label as Label>::Tokenized>;
}

impl<T: Label> Sample for Text<T> {
    type Label = T;
    fn tokenize(self, tokenizer: &tokenizer::Tokenizer) -> TokenizedText<T::Tokenized> {
        let tokens = tokenizer
            .encode(self.text, false)
            .expect("Failed to tokenize");
        let label = self.label.tokenize(&tokens, 0);
        TokenizedText {
            encoding: Encoding::from_tokenizer_encoding(
                tokens,
                tokenizer.get_padding().map_or(0, |pad| pad.pad_id),
            ),
            label: label,
        }
    }
}

impl<T: Label> Sample for TextPair<T> {
    type Label = T;
    fn tokenize(self, tokenizer: &tokenizer::Tokenizer) -> TokenizedText<T::Tokenized> {
        let tokens = tokenizer
            .encode(self.text, false)
            .expect("Failed to tokenize");
        let label = self.label.tokenize(&tokens, 0);
        TokenizedText {
            encoding: Encoding::from_tokenizer_encoding(
                tokens,
                tokenizer.get_padding().map_or(0, |pad| pad.pad_id),
            ),
            label: label,
        }
    }
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

pub trait BatchLabel: ToPyObjectConsume {}

pub struct BatchEncoding {
    input_ids: Array2<u32>,
    pad_token: u32,
}
pub struct Batch<T: BatchLabel> {
    encoding: BatchEncoding,
    labels: T,
}

pub struct Tokenizer<T: Node<Output: Sample>> {
    loader: T,
    tokenizer: tokenizer::Tokenizer,
}

impl<T: Node<Output: Sample>> Tokenizer<T> {
    fn new<S: AsRef<str>>(loader: T, tokenizer: S) -> Result<Tokenizer<T>, tokenizer::Error> {
        let tokenizer = tokenizer::Tokenizer::from_pretrained(tokenizer, None)?;
        Ok(Tokenizer { loader, tokenizer })
    }
}

impl<T: Node<Output: Sample>> Node for Tokenizer<T> {
    type Output = TokenizedText<<<<T as Node>::Output as Sample>::Label as Label>::Tokenized>;
    fn get(&self, index: usize) -> Option<Self::Output> {
        self.loader
            .get(index)
            .map(|sample| sample.tokenize(&self.tokenizer))
    }
    fn len(&self) -> Option<usize> {
        self.loader.len()
    }
    fn next(&mut self) -> Option<Self::Output> {
        self.loader
            .next()
            .map(|sample| sample.tokenize(&self.tokenizer))
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

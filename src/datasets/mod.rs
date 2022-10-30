use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

use crate::{NoLabel, Node, Span, Text, TextPair};

pub struct TxtLoader {
    lines: io::Lines<io::BufReader<File>>,
}

impl TxtLoader {
    pub fn new<P: AsRef<Path>>(file: P) -> io::Result<TxtLoader> {
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

#[derive(Serialize, Deserialize)]
struct SQuADAnswer {
    answer_start: usize,
    text: String,
}
#[derive(Serialize, Deserialize)]
struct SQuADQuestion {
    id: Option<String>,
    question: String,
    answers: Vec<SQuADAnswer>,
    is_impossible: bool,
}
#[derive(Serialize, Deserialize)]
struct SQuADParagraph {
    context: String,
    qas: Vec<SQuADQuestion>,
}
#[derive(Serialize, Deserialize)]
struct SQuADTopic {
    title: String,
    paragraphs: Vec<SQuADParagraph>,
}
#[derive(Serialize, Deserialize)]
struct SQuAD {
    data: Vec<SQuADTopic>,
}

pub struct SQuADLoader {
    texts: Vec<TextPair<Span>>,
    current_index: usize,
}

impl SQuADLoader {
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = io::BufReader::new(file);
        let squad: SQuAD = serde_json::from_reader(reader)?;
        let mut texts = Vec::new();
        for topic in squad.data {
            for paragraph in topic.paragraphs {
                let SQuADParagraph { context, qas } = paragraph;
                for qa in qas {
                    let SQuADQuestion {
                        question,
                        answers,
                        is_impossible,
                        ..
                    } = qa;
                    let mut span = None;
                    for answer in answers {
                        let answer_start = answer.answer_start;
                        let text = answer.text;
                        if !is_impossible {
                            span = Some((answer_start, answer_start + text.len() - 1));
                        }
                    }
                    let text = TextPair {
                        text: (question, context.clone()),
                        label: Span(span),
                    };
                    texts.push(text);
                }
            }
        }
        Ok(SQuADLoader {
            texts,
            current_index: 0,
        })
    }
}

impl Node for SQuADLoader {
    type Output = TextPair<Span>;
    fn get(&self, index: usize) -> Option<Self::Output> {
        let text = self.texts.get(index)?;
        Some(text.clone())
    }
    fn len(&self) -> Option<usize> {
        Some(self.texts.len())
    }
    fn next(&mut self) -> Option<Self::Output> {
        let text = self.texts.get(self.current_index)?;
        self.current_index += 1;
        Some(text.clone())
    }
}

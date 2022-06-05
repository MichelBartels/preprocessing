#[cfg(test)]
mod tests {
    use crate::Node;

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
    #[test]
    fn integration_test() {
        let txt_loader = crate::TxtLoader::new("test.txt").unwrap();
        let plain_tokenizer =
            crate::Tokenizer::new(Box::new(txt_loader), "bert-base-uncased").unwrap();
        let mut static_batcher =
            crate::StaticBatcher::new(Box::new(plain_tokenizer), 3, 32).unwrap();
        while let Some(batch) = static_batcher.next() {
            println!("{:?}", batch.encoding.input_ids);
        }
    }
}

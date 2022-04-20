// Configuration for a named entity recognization model based on:
//   Peters, Matthew E. et al. “Deep contextualized word representations.” NAACL-HLT (2018).


local bert_name = "/home/data/embedding/bert-base-cased";
// local bert_name = "/home/data/embedding/bert-base-chinese";
local max_length = 512;
local tokenizer_kwargs = {
    do_lower_case: false
};
local data_prefix = "data/mpqa2/en";

{
  "dataset_reader": {
    type: "mpqa2_exp",
    token_indexers: {
        bert: {
            type: "pretrained_transformer_mismatched",
            model_name: bert_name,
            max_length: max_length,
            tokenizer_kwargs: tokenizer_kwargs
        }
    },
  },
  "train_data_path": data_prefix + ".{}.train.txt",
  "validation_data_path": data_prefix + ".{}.dev.txt",
  "test_data_path": data_prefix + ".{}.test.txt",
  evaluate_on_test: true,
  "model": {
    "type": "crf_tagger",
    "label_encoding": "BIO",
    "constrain_crf_decoding": true,
    "calculate_span_f1": true,
    "dropout": 0.5,
    "include_start_end_transitions": false,
    "text_field_embedder": {
      "token_embedders": {
        bert: {
            type: "adapter_transformer_mismatched",
            model_name: bert_name,
            max_length: max_length,
            tokenizer_kwargs: tokenizer_kwargs,
            adapter_layers: 12,
            adapter_kwargs: {
                adapter_size: 128,
                bias: true,
            },
        }
      }
    },
    "encoder": {
        "type": "lstm",
        "input_size": 768,
        "hidden_size": 200,
        "num_layers": 1,
        "dropout": 0.5,
        "bidirectional": true
    },
  },
  "data_loader": {
    "batch_size": 64
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 0.001
    },
    "checkpointer": {
        "num_serialized_models_to_keep": 2,
    },
    "validation_metric": "+f1-measure-overall",
    "num_epochs": 25,
    "grad_norm": 5.0,
    "patience": 5,
  }
}

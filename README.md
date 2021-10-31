# Knowledge Enriched Multimodal Variational Autoencoder

## Data Preprocessing

Convert ConceptNet
```
python -m kmvae.kgnn.scripts.convert_conceptnet \
--conceptnet_csv_path 'data/conceptnet-assertions-5.7.0.csv' \
--output_path 'data/conceptnet/conceptnet' \
--language 'en'
```

Convert GoEmotions
```
python -m kmvae.scripts.preprocess.convert_goemotions \
--goemotions_input_path 'data/goemotions/data/train.tsv' \
--goemotions_output_path 'data/goemotions/data/train.json' \
--goemotions_emotion_ids_path 'data/goemotions/data/emotions.txt' \
--goemotions_ekman_mapping_path 'data/goemotions/data/ekman_mapping.json' \
&& \
python -m kmvae.scripts.preprocess.convert_goemotions \
--goemotions_input_path 'data/goemotions/data/dev.tsv' \
--goemotions_output_path 'data/goemotions/data/val.json' \
--goemotions_emotion_ids_path 'data/goemotions/data/emotions.txt' \
--goemotions_ekman_mapping_path 'data/goemotions/data/ekman_mapping.json' \
&& \
python -m kmvae.scripts.preprocess.convert_goemotions \
--goemotions_input_path 'data/goemotions/data/test.tsv' \
--goemotions_output_path 'data/goemotions/data/test.json' \
--goemotions_emotion_ids_path 'data/goemotions/data/emotions.txt' \
--goemotions_ekman_mapping_path 'data/goemotions/data/ekman_mapping.json'
```

Tokenize GoEmotions
```
python -m kmvae.scripts.preprocess.tokenize_sentences \
--load_sentences_path 'data/goemotions/data/train.json' \
--save_sentences_path 'data/goemotions/data/train.json' \
&& \
python -m kmvae.scripts.preprocess.tokenize_sentences \
--load_sentences_path 'data/goemotions/data/val.json' \
--save_sentences_path 'data/goemotions/data/val.json' \
&& \
python -m kmvae.scripts.preprocess.tokenize_sentences \
--load_sentences_path 'data/goemotions/data/test.json' \
--save_sentences_path 'data/goemotions/data/test.json'
```

Add ConceptNet entities to GoEmotions sentences
```
python -m kmvae.scripts.preprocess.add_entities_to_sentences \
--knowledge_graph_path 'data/conceptnet/conceptnet' \
--load_sentences_path 'data/goemotions/data/train.json' \
--save_sentences_path 'data/goemotions/data/train.json' \
&& \
python -m kmvae.scripts.preprocess.add_entities_to_sentences \
--knowledge_graph_path 'data/conceptnet/conceptnet' \
--load_sentences_path 'data/goemotions/data/val.json' \
--save_sentences_path 'data/goemotions/data/val.json' \
&& \
python -m kmvae.scripts.preprocess.add_entities_to_sentences \
--knowledge_graph_path 'data/conceptnet/conceptnet' \
--load_sentences_path 'data/goemotions/data/test.json' \
--save_sentences_path 'data/goemotions/data/test.json'
```

Create Emotional Context Knowledge Graph
```
python -m kmvae.scripts.preprocess.create_emotional_context_knowledge_graph \
--knowledge_graph_path 'data/conceptnet/conceptnet' \
--nrc_valence_path 'data/nrc-vad/v-scores.txt' \
--nrc_arousal_path 'data/nrc-vad/a-scores.txt' \
--nrc_dominance_path 'data/nrc-vad/d-scores.txt' \
--sentences_train_path 'data/goemotions/data/train.json' \
--sentences_val_path 'data/goemotions/data/val.json' \
--sentences_test_path 'data/goemotions/data/test.json' \
--sentences_train_output_path 'data/goemotions/data/train.json' \
--sentences_val_output_path 'data/goemotions/data/val.json' \
--sentences_test_output_path 'data/goemotions/data/test.json' \
--output_path 'data/emotional-context/emotional-context' \
--output_train_path 'data/emotional-context/train' \
--output_val_path 'data/emotional-context/val' \
--output_test_path 'data/emotional-context/test' \
--synonym_relation 'Synonym' \
--neighborhood_limit 10 \
--split True \
--special_entities '[pad]' '[context]' \
--special_relations '[pad]' '[context]' '[neighbor_context]' '[identity]' \
--train_split 0.8 \
--val_split 0.1 \
--random_seed 0
```

## Train Models

Pre-train knowledge graph models
```
python -m kmvae.kgnn.train.transe
python -m kmvae.kgnn.train.kbgat_conve_ae
```

Convert and pre-train knowledge graph VAE models
```
python -m kmvae.kgnn.scripts.convert_kbgat_encoder \
--knowledge_graph_path 'data/emotional-context/emotional-context' \
--pretrained_kbgat_path 'models_200/kbgat_ae_new.pt' \
--kbgat_encoder_path 'models_200/kbgat_encoder_new' \
--embedding_size 200 \
--latent_size 512

python -m kmvae.kgnn.scripts.convert_conve_decoder \
--knowledge_graph_path 'data/emotional-context/emotional-context' \
--conve_decoder_path 'models_200/conve_decoder_new' \
--latent_size 512 \
--embedding_size 200 \
--num_filters 32

python -m kmvae.scripts.pretrain_graph_vae
```

Convert and pre-train Optimus language VAE model
```
python -m kmvae.scripts.convert_optimus
python -m kmvae.scripts.pretrain_language_vae
```

Pre-train label VAE model
```
python -m kmvae.scripts.pretrain_label_vae \
--num_labels 28 \
--hidden_size 1024 \
--latent_size 512 \
--dropout_rate 0.1 \
--label_smoothing 0.1 \
--kl_free_bits 0.1 \
--num_steps 0 \
--label_encoder_path 'models/label_encoder_new' \
--label_decoder_path 'models/label_decoder_new'
```

Train knowledge MVAE and mmJSD models
```
python -m kmvae.scripts.train_mvae
python -m kmvae.scripts.train_mmjsd
```

## Evaluate Models

Evaluate latent classifier models
```
python -m kmvae.scripts.eval_graph_latent
python -m kmvae.scripts.eval_text_latent
```

Evaluate knowledge MVAE and mmJSD models
```
python -m kmvae.scripts.eval_mvae
python -m kmvae.scripts.eval_mmjsd
```

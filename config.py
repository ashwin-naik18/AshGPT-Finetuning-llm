BLOCK_SIZE = 512

DATASET_NAME = "teknium/OpenHermes-2.5"

TRAIN_RATIO = 0.95

SEED = 42

SHARD_SIZE = 2000

MIN_TRAINABLE_TOKENS = 1

CHUNK_SIZE = 10000

EMBEDDING_DIM = 64

NUM_LAYER = 4

NUM_HEAD = 4

BATCH_SIZE = 32          

EPOCH = 5

STEPS_PER_SHARD = SHARD_SIZE // BATCH_SIZE

TRAIN_STEP_PER_CHUNK = STEPS_PER_SHARD  

LEARNING_RATE = 3e-4

EVAL_STEPS = 20

save_dir = "/content/drive/MyDrive/SimpleStoriesChunks"

TRAIN_DIR = "/content/drive/MyDrive/openhermes_chunks/train"

VAL_DIR = "/content/drive/MyDrive/openhermes_chunks/val"
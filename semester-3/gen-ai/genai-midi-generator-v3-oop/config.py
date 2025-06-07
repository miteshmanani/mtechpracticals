class Config:
    def __init__(self):
        self.latent_dim = 256
        self.note_dim = 128
        self.time_steps = 32  # 32
        self.num_keys = 24
        self.batch_size = 64  # 64
        self.epochs = 5000  # 5000
        self.n_critic = 5
        self.lr = 0.0001
        self.beta1 = 0.5
        self.beta2 = 0.9
        self.gradient_penalty_weight = 10
        self.output_dir = "C:/mtechpracticals/semester-3/gen-ai/genai-midi-generator-v3-oop/output"
        self.data_dir = "C:/mtechpracticals/semester-3/gen-ai/genai-midi-generator-v3-oop/data"
        self.max_files = 1000

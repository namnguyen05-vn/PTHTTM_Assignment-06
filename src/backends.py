"""Equivalent Simple RNN architecture, two independent backend adapters."""
import os
import numpy as np
from .paths import ROOT
from .initialization import initial_weights

def pytorch_runtime(seed, cpu=False):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    return torch, "cuda" if torch.cuda.is_available() and not cpu else "cpu"

def keras_runtime(seed, cpu=False):
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    cuda=os.environ.get("ASG06_CUDA_DIR","E:/PTHTTM/ASG_04/ASG_04_runtime/cuda/Library/bin")
    if os.name=="nt" and os.path.isdir(cuda):
        os.environ["PATH"]=cuda+os.pathsep+os.environ["PATH"]
        global DLL_HANDLE
        DLL_HANDLE=os.add_dll_directory(cuda)
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu,True)
    return tf

def build_pytorch(torch, input_dim, hidden, output_dim):
    class SimpleRNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn=torch.nn.RNN(input_dim,hidden,batch_first=True,nonlinearity="tanh")
            self.rnn.bias_hh_l0.requires_grad_(False)
            self.head=torch.nn.Linear(hidden,output_dim)
        def forward(self,x,lengths):
            packed=torch.nn.utils.rnn.pack_padded_sequence(
                x,lengths.cpu(),batch_first=True,enforce_sorted=False)
            _,state=self.rnn(packed)
            return self.head(state[-1])
    return SimpleRNN()

def build_keras(tf, input_dim, hidden, output_dim, length):
    x=tf.keras.Input((length,input_dim),name="sequence")
    sizes=tf.keras.Input((),dtype="int32",name="length")
    mask=tf.sequence_mask(sizes,maxlen=length)
    h=tf.keras.layers.SimpleRNN(hidden,activation="tanh",name="rnn")(x,mask=mask)
    output=tf.keras.layers.Dense(output_dim,name="head")(h)
    return tf.keras.Model([x,sizes],output)

class TorchAdapter:
    def __init__(self,config,metadata,cpu=False):
        self.torch,self.device=pytorch_runtime(config["seed"],cpu)
        torch=self.torch
        out=3 if metadata["dataset"]=="retailrocket" else 1
        self.model=build_pytorch(torch,metadata["input_dim"],config["hidden_size"],out).to(self.device)
        w=initial_weights(metadata["input_dim"],config["hidden_size"],out,config["seed"])
        with torch.no_grad():
            for parameter,value in [(self.model.rnn.weight_ih_l0,w["wx"].T),
                (self.model.rnn.weight_hh_l0,w["wh"].T),(self.model.rnn.bias_ih_l0,w["bh"]),
                (self.model.rnn.bias_hh_l0,w["bh"]*0),(self.model.head.weight,w["wy"].T),
                (self.model.head.bias,w["by"])]:
                parameter.copy_(torch.tensor(value,device=self.device))
        self.optimizer=torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad],
            lr=config["learning_rate"],eps=config["adam_epsilon"])
        self.clip=config["clip_norm"]
        self.classification=out==3
        self.weights=torch.tensor(config["class_weights"],dtype=torch.float32,device=self.device)
        self.version=torch.__version__
        self.trainable=sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    def predict(self,x,lengths):
        torch=self.torch
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.as_tensor(x,device=self.device),
                torch.as_tensor(lengths)).cpu().numpy()
    def train_batch(self,x,lengths,y):
        torch=self.torch
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output=self.model(torch.as_tensor(x,device=self.device),torch.as_tensor(lengths))
        target=torch.as_tensor(y,device=self.device)
        if self.classification:
            losses=torch.nn.functional.cross_entropy(output,target.long(),reduction="none")
            loss=(losses*self.weights[target.long()]).mean()
        else:
            loss=((output[:,0]-target.float())**2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],self.clip)
        self.optimizer.step()
        return float(loss.detach())
    def save(self,prefix):
        torch=self.torch
        path=str(prefix)+".pt"
        payload=dict(model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        torch.save(payload,path+".part")
        os.replace(path+".part",path)
    def restore(self,prefix):
        torch=self.torch
        state=torch.load(str(prefix)+".pt",map_location=self.device,weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"].cpu())
        if self.device=="cuda":
            torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng"]])

class KerasAdapter:
    def __init__(self,config,metadata,cpu=False):
        self.tf=tf=keras_runtime(config["seed"],cpu)
        out=3 if metadata["dataset"]=="retailrocket" else 1
        self.model=build_keras(tf,metadata["input_dim"],config["hidden_size"],out,metadata["sequence_length"])
        w=initial_weights(metadata["input_dim"],config["hidden_size"],out,config["seed"])
        self.model.get_layer("rnn").set_weights([w["wx"],w["wh"],w["bh"]])
        self.model.get_layer("head").set_weights([w["wy"],w["by"]])
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"],epsilon=config["adam_epsilon"])
        self.optimizer.apply_gradients([(tf.zeros_like(v),v) for v in self.model.trainable_variables])
        self.optimizer.iterations.assign(0)
        self.checkpoint=tf.train.Checkpoint(model=self.model,optimizer=self.optimizer)
        self.version=tf.__version__
        self.device="cuda" if tf.config.list_physical_devices("GPU") else "cpu"
        self.trainable=int(sum(np.prod(v.shape) for v in self.model.trainable_variables))
        weights=tf.constant(config["class_weights"],dtype=tf.float32)
        @tf.function(reduce_retracing=True)
        def step(x,lengths,y):
            with tf.GradientTape() as tape:
                output=self.model([x,lengths],training=True)
                if out==3:
                    labels=tf.one_hot(tf.cast(y,tf.int32),3)
                    losses=tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=output)
                    loss=tf.reduce_mean(losses*tf.gather(weights,tf.cast(y,tf.int32)))
                else:
                    loss=tf.reduce_mean(tf.square(output[:,0]-tf.cast(y,tf.float32)))
            gradients=tape.gradient(loss,self.model.trainable_variables)
            gradients,_=tf.clip_by_global_norm(gradients,config["clip_norm"])
            self.optimizer.apply_gradients(zip(gradients,self.model.trainable_variables))
            return loss
        self.step=step
        @tf.function(reduce_retracing=True)
        def predict(x,lengths):
            return self.model([x,lengths],training=False)
        self.infer=predict
    def predict(self,x,lengths):
        return self.infer(x,lengths).numpy()
    def train_batch(self,x,lengths,y):
        return float(self.step(x,lengths,y))
    def save(self,prefix):
        self.checkpoint.write(str(prefix))
    def restore(self,prefix):
        self.checkpoint.read(str(prefix)).assert_consumed()

def create_adapter(framework,config,metadata,cpu=False):
    cls=TorchAdapter if framework=="pytorch" else KerasAdapter
    return cls(config,metadata,cpu)

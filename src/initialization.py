"""One shared initialization for algebraically equivalent RNNs."""
import numpy as np

def initial_weights(input_dim, hidden, output_dim, seed):
    rng=np.random.default_rng(seed)
    radius=np.sqrt(6/(input_dim+hidden))
    wx=rng.uniform(-radius,radius,(input_dim,hidden))
    q,r=np.linalg.qr(rng.normal(size=(hidden,hidden)))
    wh=q*np.sign(np.diag(r))
    radius=np.sqrt(6/(hidden+output_dim))
    wy=rng.uniform(-radius,radius,(hidden,output_dim))
    return {k:np.asarray(v,dtype="float32") for k,v in
            dict(wx=wx,wh=wh,bh=np.zeros(hidden),wy=wy,by=np.zeros(output_dim)).items()}

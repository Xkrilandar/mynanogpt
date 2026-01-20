imports

device
max iters 
eval interval
eval iters
batch size 
block size

class gpt config

class gpt
    def init
        super init
        self.transformer = moduledict(
            dict(
                token embedding (vocab size, embedding size)
                position embedding (block size, embedding size)
                dropout
                blocks modulelist(
                    block for _ in range n_layer
                )
                layernorm (embedding size)
            )
        )
        self.head = linear #language modeling head so (embedding size, vocab_size)
        self.transformer.wte.weight=self.head.weight #weight tying
        self.apply(self.init weights)

        for 
            init weights of c proj as normal
        

    def forward ( idx, targets)
        - B is batch, T is time, C is channel
        - B is batch, T is sequence length, C is embeddings
        idx is (B,T)
        targets is (B,T) too
        returns logits is (B,T,vocab_size)
        -
        b, t = idx.size()
        token_embeddings = self.transformer.token_embeddings(idx)
        positional_embeddings = arange(0, t, device) (T,)
        embeddings = self.tok_embds + self.pos_embds[None,:,:] (B,T,C)
        x = dropout(embeddings) (B,T,C)
        for block in self.transformer.blocks
            x = block(x)
        x = self.transformer.layer_norm(x) 
        logits = self.head(x) (B,T,vocab)
        loss = None
        if targets is not None 
            loss = cross entropy(logits.permute(0,2,1),targets) #why permute: flatten for cross entropy
        return logits, loss

    @torch no grad
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None)
        -
        return idx + new tokens predicted
        -
        for _ in range(max_new_tokens):
            logits, loss = self.forward(idx)
            logits = logits[:,-1,:] / max(temperature, 1e-8)
            if top_k is not None:
                logits = topk(logits,top_k)
            probs= softmax(logits,-1)
            next_token= multinomial(probs,1)
            idx = cat([idx,next_token], dim=1)


class causal self attention 

    def init 
        self.n_head = config._
        self.head_size = config.n_embd // config.n_head
        
        self.QKV = linear layer (n_embd, 3*n_embd)
        self.projection = linear layer (n_embd, n_embd)
        self.dropout = dropout
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size,config.block_size))

    def forward(self, x)
        B, T, C = x.size()





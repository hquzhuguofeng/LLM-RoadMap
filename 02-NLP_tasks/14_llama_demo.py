


from transformers.models.llama import LlamaConfig, LlamaModel
import torch





def run_llama():
    llamaconfig = LlamaConfig(
        vocab_size=32000,
        hidden_size = 4096 // 2,
        intermediate_size=11008 // 2,
        num_hidden_layers=32//2,
        max_position_embeddings = 2048 // 2
    )

    llama_model = LlamaModel(llamaconfig)

    input_ids = torch.randint(low=0, high=llamaconfig.vocab_size,
                              size=(8, 128))
    
    res = llama_model(input_ids)
    print(res[0].shape)

if __name__ == '__main__':
    run_llama()
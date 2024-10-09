import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fms_extras.models.speculator import MLPSpeculator
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models
def load_models(model_path, speculator_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
    speculator = MLPSpeculator.from_pretrained(speculator_path).cuda()
    return tokenizer, model, speculator

# Speculative decoding function
def speculative_decode(prompt, model, speculator, tokenizer, max_new_tokens=20, top_k=4):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    accepted_tokens = 0
    total_tokens = 0
    
    try:
        with torch.no_grad():
            while input_ids.shape[1] < max_new_tokens + len(prompt):
                # Get the last token's embedding
                last_token_embed = model.get_input_embeddings()(input_ids[:, -1:])
                
                # Generate speculative tokens
                spec_tokens = speculator(last_token_embed, num_tokens=4)
                spec_tokens = spec_tokens.topk(top_k).indices
                
                # Evaluate speculative tokens
                model_output = model(input_ids)
                model_logits = model_output.logits[:, -1:, :]
                model_tokens = model_logits.topk(top_k).indices
                
                # Compare and accept tokens
                for i in range(spec_tokens.shape[1]):
                    if spec_tokens[0, i] == model_tokens[0, 0, i]:
                        input_ids = torch.cat([input_ids, spec_tokens[:, i:i+1]], dim=-1)
                        accepted_tokens += 1
                        total_tokens += 1
                    else:
                        input_ids = torch.cat([input_ids, model_tokens[:, 0, i:i+1]], dim=-1)
                        total_tokens += 1
                        break
                
                if input_ids[0, -1] == tokenizer.eos_token_id:
                    break
        
        logger.info(f"Accepted tokens: {accepted_tokens}, Total tokens: {total_tokens}")
        return tokenizer.decode(input_ids[0])
    
    except Exception as e:
        logger.error(f"Error in speculative decoding: {str(e)}")
        return prompt  # Return original prompt in case of error

# Main execution
if __name__ == "__main__":
    model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    speculator_path = "ibm-fms/llama3-8b-accelerator"
    
    try:
        tokenizer, model, speculator = load_models(model_path, speculator_path)
        
        prompt = "Explain the concept of machine learning in simple terms:"
        result = speculative_decode(prompt, model, speculator, tokenizer)
        print(f"Generated text:\n{result}")
    
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
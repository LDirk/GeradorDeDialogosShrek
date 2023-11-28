import torch
from utils import falas_burro
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
from nltk.metrics import jaccard_distance
from nltk.translate.bleu_score import sentence_bleu
from sacremoses import MosesDetokenizer

def donkey():
    dialogs = falas_burro()
    return dialogs

def treinamento_donkey(epochs, lr=1e-5):

    dialogs = donkey()

    # Carregando o modelo pré-treinado GPT
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Tokenizando os diálogos
    tokenized_dialogs = [tokenizer.encode(dialog, return_tensors="pt") for dialog in dialogs]

    # Configuração de Fine-tuning
    optimizer = torch.optim.Adam(model.parameters(), lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # epochs = 500
    for epoch in range(epochs):
        for tokenized_dialog in tokenized_dialogs:
            optimizer.zero_grad()
            #outputs = model(**{"input_ids": tokenized_dialog})
            outputs = model(input_ids=torch.tensor(tokenized_dialog).unsqueeze(0))

            logits = outputs.logits
            #loss = loss_fn(logits.view(-1, logits.shape[-1]), tokenized_dialog.view(-1))
            loss = loss_fn(logits.view(-1, logits.shape[-1]), torch.tensor(tokenized_dialog).view(-1))

            loss.backward()
            #print(epoch)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
    
    # Caminho para salvar o modelo treinado
    output_dir = f"gpt2_finetuned_model_{epochs}"
        
    # Cria o diretório se não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Salva o modelo e os parâmetros do otimizador
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer_state_dict.pth"))

    return f"Treinamento concluído. Modelo salvo em {output_dir}."

# model_path é o caminho do modelo treinado, exemplo: model_path = 'gpt2_finetuned_model_1000'
# prompt entrada de texto, exemplo: prompt ="what is your nome ?"

def prompt_donkey(prompt, model_path, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_p=0.90, do_sample=True, temperature=0.2):

    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    # Tokenizar o texto de entrada
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    output = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        top_p=top_p,
        do_sample=do_sample,
        temperature=temperature,
        pad_token_id=model.config.eos_token_id,
    )

    # Decodificar o texto gerado
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return(print(generated_text))

# Função para calcular a similaridade de Jaccard
def jaccard_similarity(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    return 1 - jaccard_distance(a, b)

# Função para calcular o BLEU Score
def calculate_bleu(reference, candidate):
    # Convertendo as strings em listas de tokens
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()

    # Calculando o BLEU Score
    bleu_score = sentence_bleu(reference_tokens, candidate_tokens)
    return bleu_score

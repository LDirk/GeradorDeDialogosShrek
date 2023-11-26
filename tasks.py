import torch
from utils import falas_burro
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

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

def prompt_donkey(prompt,model_path):

    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    # Tokenizar o texto de entrada
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Gerar texto
    output = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,
    no_repeat_ngram_size=2,
    top_p=0.95,
    do_sample=True,  # Alterado para True
    temperature = 0.5,
    pad_token_id=model.config.eos_token_id,  # Definindo o token de preenchimento para o token de fim de sequência (eos_token_id)
    )

    # Decodificar o texto gerado
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return(generated_text)

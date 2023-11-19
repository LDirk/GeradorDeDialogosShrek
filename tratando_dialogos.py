import PyPDF2

def extrair_dialogos(script):
    dialogos = []
    dialogo_atual = {'falas': []}

    for linha in script.split('\n'):
        linha = linha.strip()
        if ':' in linha:
            if dialogo_atual['falas']:
                dialogos.append(dialogo_atual)
                dialogo_atual = {'falas': []}
            dialogo_atual['falas'].append(linha + ' ')

    if dialogo_atual['falas']:
        dialogos.append(dialogo_atual)

    return dialogos

caminho_arquivo = '/content/pdfcoffee.com_shrek-roteiro-4-pdf-free.pdf'

with open(caminho_arquivo, 'rb') as arquivo:  # 'rb' para abrir em modo binário
    leitor_pdf = PyPDF2.PdfReader(arquivo)
    numero_paginas = len(leitor_pdf.pages)

    script = ''
    for pagina_num in range(numero_paginas):
        pagina = leitor_pdf.pages[pagina_num]
        script += pagina.extract_text()

# Adicionando aspas no início e no final de cada diálogo, com vírgula após as aspas
total_dialogos = len(extrair_dialogos(script))
for i, dialogo in enumerate(extrair_dialogos(script), start=1):
    falas_formatadas = ' '.join(dialogo["falas"]).strip()
    print('"{}",'.format( falas_formatadas))

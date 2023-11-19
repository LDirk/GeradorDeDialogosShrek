# pip install PyPDF2
import re
import PyPDF2

# O código considera que a cada ":" surge um novo dialogo.


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

# Caminho do arquivo
caminho_arquivo = '/content/pdfcoffee.com_shrek-roteiro-4-pdf-free.pdf'

with open(caminho_arquivo, 'rb') as arquivo:  # 'rb' para abrir em modo binário
    leitor_pdf = PyPDF2.PdfReader(arquivo)
    numero_paginas = len(leitor_pdf.pages)

    script = ''
    for pagina_num in range(numero_paginas):
        pagina = leitor_pdf.pages[pagina_num]
        script += pagina.extract_text()

lista_dialogos = extrair_dialogos(script)

for i, dialogo in enumerate(lista_dialogos, start=1):
    print('Diálogo {}: {}'.format(i, "".join(dialogo["falas"])))

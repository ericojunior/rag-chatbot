# Como subir para o GitHub

O repositório remoto já está configurado como:

`https://github.com/ericojunior/rag-chatbot.git`

## Opção A (recomendada): SSH

1. Garanta que você tem uma chave SSH e que ela está cadastrada no GitHub.
2. Troque a URL do remote para SSH:

```bash
git remote set-url origin git@github.com:ericojunior/rag-chatbot.git
git push -u origin main
```

## Opção B: HTTPS com Personal Access Token (PAT)

1. Gere um token no GitHub (Settings → Developer settings → Personal access tokens).
2. Faça o push e quando pedir senha, use o **token** (não a senha da conta):

```bash
git push -u origin main
```


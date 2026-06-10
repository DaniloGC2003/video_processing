import os
import subprocess

# Caminho da pasta raiz
PASTA_RAIZ = r".\videos_flexao"

# Extensões de vídeo aceitas
EXTENSOES = (".mp4", ".avi", ".mov", ".mkv", ".webm")

def obter_resolucao(video):
    comando = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        video
    ]

    resultado = subprocess.run(
        comando,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    largura, altura = map(int, resultado.stdout.strip().split("x"))
    return largura, altura

def tornar_quadrado(video_entrada, video_saida):
    largura, altura = obter_resolucao(video_entrada)

    tamanho = max(largura, altura)

    # Centraliza o vídeo e adiciona padding preto
    filtro = (
        f"pad={tamanho}:{tamanho}:"
        f"(ow-iw)/2:(oh-ih)/2:black"
    )

    comando = [
        "ffmpeg",
        "-i", video_entrada,
        "-vf", filtro,
        "-c:a", "copy",
        video_saida
    ]

    subprocess.run(comando)

def processar_videos():
    for raiz, _, arquivos in os.walk(PASTA_RAIZ):
        for arquivo in arquivos:
            if arquivo.lower().endswith(EXTENSOES):

                caminho_entrada = os.path.join(raiz, arquivo)

                nome, ext = os.path.splitext(arquivo)

                caminho_saida = os.path.join(
                    raiz,
                    f"{nome}_quadrado{ext}"
                )

                print(f"Processando: {caminho_entrada}")

                try:
                    tornar_quadrado(caminho_entrada, caminho_saida)
                    print(f"Salvo em: {caminho_saida}\n")

                except Exception as e:
                    print(f"Erro em {arquivo}: {e}\n")

if __name__ == "__main__":
    processar_videos()
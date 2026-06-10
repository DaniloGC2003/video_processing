import tensorflow as tf
import numpy as np
import cv2 as cv
import os
import math

SENSITIVITY = 100
TRUST_THRESHOLD = 0.50

def calcular_angulo(A, B, C):
    """
    Calcula o ângulo ABC (em graus)
    A, B, C são arrays [y, x, score]
    """
    BA = np.array([A[0] - B[0], A[1] - B[1]])
    BC = np.array([C[0] - B[0], C[1] - B[1]])

    produto = np.dot(BA, BC)
    mag_BA = np.linalg.norm(BA)
    mag_BC = np.linalg.norm(BC)

    if mag_BA == 0 or mag_BC == 0:
        return None

    cos = produto / (mag_BA * mag_BC)
    cos = np.clip(cos, -1.0, 1.0)   # evita erros numéricos

    angulo = math.degrees(math.acos(cos))
    return angulo

diretorio_atual = os.path.dirname(os.path.realpath(__file__))

caminho_modelo_pb = os.path.join(diretorio_atual, 'models', 'movenet_thunder')
model = tf.saved_model.load(caminho_modelo_pb)
movenet = model.signatures['serving_default']


path2 = os.path.join("videos_flexao", "danilo_02_06_healthy.mp4")
#path2 = os.path.join("videos_flexao", "danilo_rehab.mp4")
cap = cv.VideoCapture(path2) ##Aqui é onde você modifica qual dos vídeos você ira processar
print("Video aberto?", cap.isOpened())

if not cap.isOpened():
    print("ERRO: Não foi possível abrir o vídeo!")

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}



def desenhar_numero(frame, numero, estado):
    fonte = cv.FONT_HERSHEY_SIMPLEX
    escala = 1
    cor = (255, 255, 255)
    espessura = 2

    posicao = (10, 30)
    posicao2 = (10, 50)
    texto = "Rep count: " + str(numero)
    texto2 = "Estado do braco: " + estado

    cv.putText(frame, texto, posicao, fonte, escala, cor, espessura)
    #cv.putText(frame, texto2, posicao2, fonte, escala, cor, espessura)

class Individuo:
    def  __init__(self, genero):
        self.genero = genero
        self.estado = "desconhecido"
        self.reto = False
        self.estado_anterior = "desconhecido"
        self.contador_de_flexoes_idividual = 0
        self.estado_pre_anterior = 0
        self.lado = "esquerdo"
        self.comprimento = 0
        self.comprimento_atual = 0
        self.menor_comprimento = 10
        self.nivel_de_ext = 0
        self.angulo_braco = 0
        self.dobrou = False  # marca se já desceu na flexão

    def verificar_estado(self, keypoints):

        # keypoints = (1,1,17,3)
        kp = keypoints[0][0]

        # Seleção dos pontos baseado no lado
        if self.lado[0] == 'd':
            ombro = kp[6]     # ombro direito
            cotovelo = kp[8]  # cotovelo direito
            punho = kp[10]    # punho direito
        else:
            ombro = kp[5]     # ombro esquerdo
            cotovelo = kp[7]  # cotovelo esquerdo
            punho = kp[9]     # punho esquerdo

        # Se os keypoints estão ruins, não calcula
        if ombro[2] < 0.3 or cotovelo[2] < 0.3 or punho[2] < 0.3:
            self.estado = "desconhecido"
            return

        # Calcula ângulo
        angulo = calcular_angulo(ombro, cotovelo, punho)
        if angulo is None:
            self.estado = "desconhecido"
            return

        # Classificação
        if angulo > 160:
            self.estado = "estendido"
        elif 70 < angulo < 110:
            self.estado = "90 graus"
        elif angulo <= 85:
            self.estado = "flexionado"
        else:
            self.estado = "parcial"

        self.angulo_braco = angulo
        return self.estado

    def postura_reta(self, kp):
        """
        kp = keypoints[0][0]  (17,3)
        Retorna True se o tronco estiver reto em relação às pernas
        """
        kp = kp[0][0]

        if self.lado[0] == 'd':  # lado direito
            ombro = kp[6]
            quadril = kp[12]
            joelho = kp[14]      # joelho direito
        else:  # esquerdo
            ombro = kp[5]
            quadril = kp[11]
            joelho = kp[13]

        # Confiança mínima
        if ombro[2] < 0.3 or quadril[2] < 0.3 or joelho[2] < 0.3:
            return False

        # Vetor TRONCO (quadril -> ombro)
        tronco = np.array([ombro[0] - quadril[0], ombro[1] - quadril[1]])

        # Vetor PERNA (quadril -> joelho)
        perna = np.array([joelho[0] - quadril[0], joelho[1] - quadril[1]])

        # Ângulo entre tronco e perna
        dot = np.dot(tronco, perna)
        norm = np.linalg.norm(tronco) * np.linalg.norm(perna)

        if norm == 0:
            return False

        cosang = dot / norm
        cosang = max(-1, min(1, cosang))  # evitar erros numéricos

        angulo = math.degrees(math.acos(cosang))

        # Tronco reto geralmente = ângulo próximo de 180° (tronco alinhado com perna)
        self.reto = angulo > 155 and angulo < 200

        if self.reto:
            print("Postura reta detectada. Ângulo tronco-perna:", angulo)
        else:
            print("Postura inadequada. Ângulo tronco-perna:", angulo)

        return self.reto

    def confere_mov(self, tempo, contflec, contelap):

        # Detectou braço flexionado (fase baixa do movimento)
        if self.estado == "90 graus":
            self.dobrou = True   # marcou que já desceu
        
        # Detectou braço estendido (voltou para cima)
        if self.estado == "estendido" and self.dobrou and self.reto:
            self.contador_de_flexoes_idividual += 1
            contflec += 1
            contelap = tempo
            self.dobrou = False  # reseta para esperar a próxima descida
        
        if self.reto == False:
            self.dobrou = False  # reseta se não estiver reto

        #print("Cont flec", contflec)
        return [contflec, contelap]

    def set_lado(self, lado):
        self.lado = "esquerdo" if lado[0] == 'e' else "direito"

    # Verifica o quanto o braço está esticado ou dobrado para identificar o limite do movimento
    def comparar_extensao(self, comp2):
        if comp2 > self.comprimento:
            self.comprimento = comp2
        if comp2 < self.menor_comprimento:
            self.menor_comprimento = comp2

    # Atualiza o objeto Pessoa com a medida atual para futuras comparações
    def set_comprimento_atual(self, c):
        self.comprimento_atual = c

    def get_estado(self):
        return self.estado

def erro_de_10(v1, v2):
    if v1 > v2:
        return v1 < v2 * 1.5
    return v1 * 1.5 > v2

def draw_keypoints(frame, keypoints, conf):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > conf:
            cv.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1)

def draw_connections(frame, keypoints, edges, conf, side):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge in edges:
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if c1 > conf and c2 > conf:
            line_color = (0,0,255)
            if edge == (12,14) or edge == (14,16) or edge == (11,13) or edge == (13,15):
                if side == "direito":
                    if edge == (12,14) or edge == (14,16):
                        line_color = (255,0,0)
                elif side == "esquerdo":
                    if edge == (11,13) or edge == (13,15):
                        line_color = (255,0,0)

            cv.line(frame, (int(x1),int(y1)), (int(x2),int(y2)), line_color, 2)

def modulo(v):
    return abs(v)

def calc_comprimento(keypoints, lado):

    # keypoints = (1,1,17,3)
    linha = keypoints[0][0]  # (17,3)

    # valida
    if linha.shape[0] != 17:
        print("ERRO: keypoints com tamanho inválido:", linha.shape)
        return 0

    # valida confiança
    if np.mean(linha[:,2]) < 0.2:
        print("Pessoa não detectada")
        return 0

    # agora pode acessar keypoints com segurança
    return modulo(linha[6][0] - linha[10][0])


number_of_reps = int(input("number of reps: "))
most_reto = input("Mostrar se esta reto? ")
Pessoa = Individuo("Masculino")
contagem_de_vezes = 0
Pessoa.set_lado(input("defina se é o lado esquerdo ou direito: "))
hip = 0
knee = 0
heel = 0
if Pessoa.lado == "esquerdo":
    hip = 11
    knee = 13
    heel = 15
elif Pessoa.lado == "direito":
    hip = 12
    knee = 14
    heel = 16
limitador_de_frames = int(input("Defina limitador de frames: "))

contador_De_flexoes = 0
contador_de_tempo_elapsado = 0
lado = Pessoa.lado

estado_anterior = "desconhecido"
estado_atual = "desconhecido"

movimento_iniciado = False
contagem_reps = 0
speeds = []
totalAbsoluteAngleTraveled = 0
previous_keypoints = None

while cap.isOpened():
    # lê próximo quadro
    # ret: booleano que indica se a leitura ocorreu com sucesso
    # frame: matriz de pixels
    ret, frame = cap.read()
    if not ret:
        break

    # NÃO redimensiona o frame original
    original_frame = frame.copy()

    # ----- Resize para movnet. Ajusta imagem -----
    # Converte formato BGR em RGB
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # Transforma imagem em array de imagens; redimensiona imagem para 256x256
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)
    # converte valores dos pixels para int32
    input_image = tf.cast(img, dtype=tf.int32)

    # ----- Inferência -----
    try:
        outputs = movenet(input=input_image)
    except:
        outputs = movenet(serving_default_input=input_image)

    #  Extrai os resultados da IA (coordenadas X, Y e a confiança de cada ponto do corpo)
    keypoints_with_scores = outputs["output_0"].numpy()
    if contagem_de_vezes == 0:
        print(keypoints_with_scores)

    angulo_quadril_joelho_tornozelo = calcular_angulo(keypoints_with_scores[0][0][hip], keypoints_with_scores[0][0][knee], keypoints_with_scores[0][0][heel])
    print(f"angulo quadril joelho tornozelo: {angulo_quadril_joelho_tornozelo}; estado do movimento: {movimento_iniciado}; "
          f"trust score: {keypoints_with_scores[0][0][hip][2]:.2f}, {keypoints_with_scores[0][0][knee][2]:.2f}, {keypoints_with_scores[0][0][heel][2]:.2f}")

    if keypoints_with_scores[0][0][hip][2] < TRUST_THRESHOLD:
        print("LOW TRUST ON HIP")
    if keypoints_with_scores[0][0][knee][2] < TRUST_THRESHOLD:
        print("LOW TRUST ON KNEE")
    if keypoints_with_scores[0][0][heel][2] < TRUST_THRESHOLD:
        print("LOW TRUST ON HEEL")

    if contagem_de_vezes >= 1:
        speed = angulo_quadril_joelho_tornozelo - calcular_angulo(previous_keypoints[0], previous_keypoints[1], previous_keypoints[2])
        speeds.append(speed)
        print(speeds[-1]) # print last element
        totalAbsoluteAngleTraveled += abs(speed)
    previous_keypoints = [keypoints_with_scores[0][0][hip], keypoints_with_scores[0][0][knee], keypoints_with_scores[0][0][heel]]

    if not movimento_iniciado and angulo_quadril_joelho_tornozelo < 100:
        movimento_iniciado = True
        print("Movimento iniciado")
    if movimento_iniciado and angulo_quadril_joelho_tornozelo > 130:
        print("Movimento finalizado")
        movimento_iniciado = False
        contagem_reps += 1

    #stop processing the video once the exercise is completed
    if contagem_reps == number_of_reps:
        break


    '''comprimento = calc_comprimento(keypoints_with_scores, lado) # calcular a distância entre pontos específicos
    Pessoa.comparar_extensao(comprimento)
    Pessoa.set_comprimento_atual(comprimento)

    estado_atual = Pessoa.verificar_estado(keypoints_with_scores)
    Pessoa.postura_reta(keypoints_with_scores)

    # valida se o movimento foi concluído sem erros
    res = Pessoa.confere_mov(
        contagem_de_vezes,
        contador_De_flexoes,
        contador_de_tempo_elapsado
    )
    # Atualiza as variáveis globais com o resultado da validação
    contador_De_flexoes, contador_de_tempo_elapsado = res'''

    # mecanismo de segurança para encerrar o programa automaticamente após processar um número definido de quadros
    if contagem_de_vezes > limitador_de_frames:
        print("Fim por limites")
        break

    # Desenha no frame ORIGINAL (sem resize)
    # Desenha as linhas coloridas conectando as articulações
    draw_connections(original_frame, keypoints_with_scores, EDGES, 0.2, Pessoa.lado)
    # Desenha os pontos (círculos) em cada articulação detectada
    draw_keypoints(original_frame, keypoints_with_scores, 0.2)
    # Escreve na tela o número atual de flexões e o estado do usuário (ex: se está "descendo")
    desenhar_numero(original_frame, contagem_reps, Pessoa.get_estado())

    # Mostra no tamanho original do vídeo; Abre a janela e mostra o vídeo sendo processado em tempo real
    cv.imshow("tela", original_frame)

    # Incrementa o contador de quadros processados
    contagem_de_vezes += 1

    # Aguarda 10 milissegundos entre os quadros e verifica se a tecla 'q' foi pressionada para fechar o programa imediatamente.
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

print("Total reps:", contagem_reps)
cap.release()
cv.destroyAllWindows()

#calculate fluidity
totaljerk = 0
for i in range(len(speeds) - 1):
    jerk = abs(speeds[i+1] - speeds[i])
    totaljerk += jerk

jerkratio = totaljerk / totalAbsoluteAngleTraveled
score = 100 - jerkratio * SENSITIVITY
print(f"Score: {score}%")

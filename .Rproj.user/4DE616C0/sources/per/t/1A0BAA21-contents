# ==============================================================================
# AULA COMPUTACIONAL: ESTIMAÇÃO INTERVALAR
# Disciplina: Inferência Estatística para Atuária (ET659)
# Unidade II: Intervalos de Confiança via Quantidade Pivotal
# ==============================================================================


# Configuração inicial

set.seed(2026)
M <- 10000   # Número de réplicas de Monte Carlo
alpha <- 0.05

cat("=== INICIANDO SIMULAÇÃO DE MONTE CARLO (", M, " RÉPLICAS) ===\n\n", sep="")

# ------------------------------------------------------------------------------
# 1. IC PARA A MÉDIA (VARIÂNCIA CONHECIDA)
# ------------------------------------------------------------------------------
n1 <- 50
mu_true <- 100
sigma_known <- 15

# Matriz onde cada coluna é uma amostra independente
X_mat_1 <- matrix(rnorm(n1 * M, mean = mu_true, sd = sigma_known), nrow = n1, ncol = M)

# Vetorização das estimativas e limites
x_bars_1 <- colMeans(X_mat_1)
z_quantil <- qnorm(1 - alpha/2)

LI_1 <- x_bars_1 - z_quantil * (sigma_known / sqrt(n1))
LS_1 <- x_bars_1 + z_quantil * (sigma_known / sqrt(n1))

# Cálculo da probabilidade de cobertura empírica
cobertura_1 <- mean(LI_1 <= mu_true & LS_1 >= mu_true)

cat("1. Média (sigma^2 conhecido) | Cobertura Nominal: 0.95 | Empírica:", cobertura_1, "\n")


# ------------------------------------------------------------------------------
# 2. IC PARA A MÉDIA (VARIÂNCIA DESCONHECIDA)
# ------------------------------------------------------------------------------
# Utilizando as mesmas amostras de X_mat_1, mas estimando S para cada coluna
S_vec <- apply(X_mat_1, 2, sd)
t_quantil <- qt(1 - alpha/2, df = n1 - 1)

LI_2 <- x_bars_1 - t_quantil * (S_vec / sqrt(n1))
LS_2 <- x_bars_1 + t_quantil * (S_vec / sqrt(n1))

cobertura_2 <- mean(LI_2 <= mu_true & LS_2 >= mu_true)

cat("2. Média (sigma^2 desconhecido) | Cobertura Nominal: 0.95 | Empírica:", cobertura_2, "\n")


# ------------------------------------------------------------------------------
# 3. IC PARA PROPORÇÃO POPULACIONAL (ASSINTÓTICO DE WALD)
# ------------------------------------------------------------------------------
n_prop <- 100
p_true <- 0.35

# Matriz de ensaios de Bernoulli
Y_mat <- matrix(rbinom(n_prop * M, size = 1, prob = p_true), nrow = n_prop, ncol = M)
p_hats <- colMeans(Y_mat)

# Erro padrão estimado para cada amostra
se_p_vec <- sqrt(p_hats * (1 - p_hats) / n_prop)

LI_3 <- p_hats - z_quantil * se_p_vec
LS_3 <- p_hats + z_quantil * se_p_vec

cobertura_3 <- mean(LI_3 <= p_true & LS_3 >= p_true)

cat("3. Proporção (Assintótico)      | Cobertura Nominal: 0.95 | Empírica:", cobertura_3, "\n")


# ------------------------------------------------------------------------------
# 4. IC PARA A RAZÃO DE VARIÂNCIAS
# ------------------------------------------------------------------------------
n_X <- 30
n_Y <- 40
sigma_X <- 10
sigma_Y <- 15
razao_verdadeira <- (sigma_X^2) / (sigma_Y^2)

# Simulação das amostras X e Y
X_mat_4 <- matrix(rnorm(n_X * M, mean = 50, sd = sigma_X), nrow = n_X, ncol = M)
Y_mat_4 <- matrix(rnorm(n_Y * M, mean = 55, sd = sigma_Y), nrow = n_Y, ncol = M)

S2_X_vec <- apply(X_mat_4, 2, var)
S2_Y_vec <- apply(Y_mat_4, 2, var)
razao_S2_vec <- S2_X_vec / S2_Y_vec

f_inf <- qf(alpha/2, df1 = n_X - 1, df2 = n_Y - 1)
f_sup <- qf(1 - alpha/2, df1 = n_X - 1, df2 = n_Y - 1)

LI_4 <- razao_S2_vec * (1 / f_sup)
LS_4 <- razao_S2_vec * (1 / f_inf)

cobertura_4 <- mean(LI_4 <= razao_verdadeira & LS_4 >= razao_verdadeira)

cat("4. Razão de Variâncias          | Cobertura Nominal: 0.95 | Empírica:", cobertura_4, "\n")


# ------------------------------------------------------------------------------
# 5. IC PARA DIFERENÇA DE MÉDIAS (VARIÂNCIAS ASSUMIDAS IGUAIS)
# Obs: Para manter a validade exata do teste T de Student, impomos sigma_X = sigma_Y
# ------------------------------------------------------------------------------
sigma_comum <- 12
mu_X <- 50
mu_Y <- 55
dif_mu_verdadeira <- mu_X - mu_Y

X_mat_5 <- matrix(rnorm(n_X * M, mean = mu_X, sd = sigma_comum), nrow = n_X, ncol = M)
Y_mat_5 <- matrix(rnorm(n_Y * M, mean = mu_Y, sd = sigma_comum), nrow = n_Y, ncol = M)

x_bars_5 <- colMeans(X_mat_5)
y_bars_5 <- colMeans(Y_mat_5)

S2_X_5 <- apply(X_mat_5, 2, var)
S2_Y_5 <- apply(Y_mat_5, 2, var)

# Variância combinada
Sp2_vec <- ((n_X - 1) * S2_X_5 + (n_Y - 1) * S2_Y_5) / (n_X + n_Y - 2)
Sp_vec <- sqrt(Sp2_vec)

t_quantil_dif <- qt(1 - alpha/2, df = n_X + n_Y - 2)
erro_padrao_dif_vec <- Sp_vec * sqrt((1 / n_X) + (1 / n_Y))

LI_5 <- (x_bars_5 - y_bars_5) - t_quantil_dif * erro_padrao_dif_vec
LS_5 <- (x_bars_5 - y_bars_5) + t_quantil_dif * erro_padrao_dif_vec

cobertura_5 <- mean(LI_5 <= dif_mu_verdadeira & LS_5 >= dif_mu_verdadeira)

cat("5. Diferença de Médias          | Cobertura Nominal: 0.95 | Empírica:", cobertura_5, "\n")

# ==============================================================================
# FIM DA SIMULAÇÃO
# ==============================================================================
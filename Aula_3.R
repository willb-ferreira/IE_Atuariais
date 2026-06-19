# ==============================================================================
# AULA COMPUTACIONAL: TESTES DE HIPÓTESES (MP, UMP e TRVG)
# Disciplina: Inferência Estatística para Atuária (ET659)
# ==============================================================================

set.seed(2026)
M <- 10000 # Número de réplicas de Monte Carlo para validação empírica

cat("=== AVALIAÇÃO COMPUTACIONAL DE TESTES DE HIPÓTESES ===\n\n")

# ------------------------------------------------------------------------------
# 1. TESTE MAIS PODEROSO (MP) - Lema de Neyman-Pearson
# Modelo: X_i ~ N(mu, 1). 
# H_0: mu = 0  vs  H_1: mu = 1 (Hipóteses Simples)
# Pelo Lema NP, a região crítica simplifica-se para: X_bar > k
# Para alpha = 0.05, sob H_0: X_bar ~ N(0, 1/n) -> k = qnorm(1-alpha, 0, 1/sqrt(n))
# ------------------------------------------------------------------------------
n1 <- 25
alpha <- 0.05
mu_0 <- 0
mu_1 <- 1

# Valor crítico exato
k_critico <- qnorm(1 - alpha, mean = mu_0, sd = 1/sqrt(n1))

# Simulação sob H_0 (Para verificar o Erro do Tipo I - tamanho do teste)
X_H0 <- matrix(rnorm(n1 * M, mean = mu_0, sd = 1), nrow = n1, ncol = M)
x_bars_H0 <- colMeans(X_H0)
erro_tipo_I_empirico <- mean(x_bars_H0 > k_critico)

# Simulação sob H_1 (Para verificar o Poder empírico do teste)
X_H1 <- matrix(rnorm(n1 * M, mean = mu_1, sd = 1), nrow = n1, ncol = M)
x_bars_H1 <- colMeans(X_H1)
poder_empirico_MP <- mean(x_bars_H1 > k_critico)
poder_teorico_MP <- 1 - pnorm(k_critico, mean = mu_1, sd = 1/sqrt(n1))

cat("1. Teste MP (Neyman-Pearson) | H0: mu=0 vs H1: mu=1\n")
cat("   Erro Tipo I (Nominal):", alpha, "| (Empírico):", erro_tipo_I_empirico, "\n")
cat("   Poder do Teste (Teórico):", round(poder_teorico_MP, 4), "| (Empírico):", poder_empirico_MP, "\n\n")


# ------------------------------------------------------------------------------
# 2. TESTE UNIFORMEMENTE MAIS PODEROSO (UMP) e FUNÇÃO PODER
# Modelo: X_i ~ Exp(lambda). Parametrização theta = 1/lambda (média).
# H_0: theta <= 2  vs  H_1: theta > 2 (Hipótese Composta Unilateral)
# Família Exponencial: A soma S = sum(X_i) é RVM. Rejeita H_0 se S > c.
# Sob H_0 (no limite theta=2), S ~ Gama(n, scale=2).
# ------------------------------------------------------------------------------
n2 <- 30
theta_0 <- 2

# Valor crítico c para a soma das exponenciais
c_critico <- qgamma(1 - alpha, shape = n2, scale = theta_0)

# Construção da Função Poder empírica pi(theta)
theta_grid <- seq(1, 5, length.out = 20)
poder_UMP <- numeric(length(theta_grid))

for(i in seq_along(theta_grid)) {
  theta_atual <- theta_grid[i]
  # Simulação de amostras exponenciais (rate = 1/theta)
  S_mat <- matrix(rexp(n2 * M, rate = 1/theta_atual), nrow = n2, ncol = M)
  somas <- colSums(S_mat)
  poder_UMP[i] <- mean(somas > c_critico)
}

cat("2. Teste UMP | H0: theta <= 2 vs H1: theta > 2\n")
cat("   A função poder pi(theta) foi calculada. Observe seu comportamento monótono.\n")
cat("   Para theta = 2 (fronteira), pi(2) = ", poder_UMP[which.min(abs(theta_grid - 2))], " (aproxima-se de alpha = 0.05)\n\n")

# Para fins visuais 
plot(theta_grid, poder_UMP, type="b", pch=19, col="blue",
     xlab=expression(theta), ylab=expression(pi(theta)),
     main="Função Poder do Teste UMP")
abline(v=2, lty=2, col="red")
abline(h=0.05, lty=2, col="red")

# ==============================================================================
# AULA COMPUTACIONAL: PROPRIEDADES ASSINTÓTICAS EM TESTES DE HIPÓTESES
# Disciplina: Inferência Estatística para Atuária (ET659)
# ==============================================================================

set.seed(2026)
M <- 10000 # Número de réplicas de Monte Carlo
alpha <- 0.05
vetor_n <- c(10, 30, 50, 100, 500)

# Estruturas para armazenamento dos resultados
res_MP   <- data.frame(n = vetor_n, Erro_Tipo_I = NA, Poder = NA)
res_UMP  <- data.frame(n = vetor_n, Nivel_Empirico = NA, Poder = NA)
res_TRVG <- data.frame(n = vetor_n, Erro_Tipo_I_Assintotico = NA, Poder = NA)

cat("=== INICIANDO SIMULAÇÕES MONTE CARLO (M =", M, ") ===\n\n")

# ------------------------------------------------------------------------------
# 3. TESTE MAIS PODEROSO (MP) - Lema de Neyman-Pearson
# Modelo: X_i ~ N(mu, 1). 
# H_0: mu = 0  vs  H_1: mu = 1
# ------------------------------------------------------------------------------
mu_0 <- 0
mu_1 <- 1

for(i in seq_along(vetor_n)) {
  n <- vetor_n[i]
  k_critico <- qnorm(1 - alpha, mean = mu_0, sd = 1/sqrt(n))
  
  # Sob H_0 (Erro do Tipo I)
  X_H0 <- matrix(rnorm(n * M, mean = mu_0, sd = 1), nrow = n, ncol = M)
  erro_I <- mean(colMeans(X_H0) > k_critico)
  
  # Sob H_1 (Poder)
  X_H1 <- matrix(rnorm(n * M, mean = mu_1, sd = 1), nrow = n, ncol = M)
  poder <- mean(colMeans(X_H1) > k_critico)
  
  res_MP[i, c("Erro_Tipo_I", "Poder")] <- c(erro_I, poder)
}

cat("1. TESTE MP (H0: mu = 0 vs H1: mu = 1)\n")
print(res_MP, row.names = FALSE)
cat("\n")

# ------------------------------------------------------------------------------
# 4. TESTE UNIFORMEMENTE MAIS PODEROSO (UMP)
# Modelo: X_i ~ Exp(lambda). Parametrização theta = 1/lambda.
# H_0: theta <= 2  vs  H_1: theta > 2
# Estatística suficiente S = sum(X_i). Rejeita-se H0 se S > c.
# ------------------------------------------------------------------------------
theta_0 <- 2   # Fronteira de H0 (para avaliar o nível máximo do teste)
theta_1 <- 3   # Ponto na região alternativa (para avaliar o poder)

for(i in seq_along(vetor_n)) {
  n <- vetor_n[i]
  # O valor crítico exato é derivado da Gama(n, escala = theta_0)
  c_critico <- qgamma(1 - alpha, shape = n, scale = theta_0)
  
  # Sob H_0 (theta = 2) - Avaliação do Tamanho do Teste
  S_H0 <- colSums(matrix(rexp(n * M, rate = 1/theta_0), nrow = n, ncol = M))
  nivel <- mean(S_H0 > c_critico)
  
  # Sob H_1 (theta = 3) - Avaliação do Poder do Teste
  S_H1 <- colSums(matrix(rexp(n * M, rate = 1/theta_1), nrow = n, ncol = M))
  poder <- mean(S_H1 > c_critico)
  
  res_UMP[i, c("Nivel_Empirico", "Poder")] <- c(nivel, poder)
}

cat("2. TESTE UMP (H0: theta <= 2 vs H1: theta > 2)\n")
print(res_UMP, row.names = FALSE)
cat("\n")

# ------------------------------------------------------------------------------
# 5. TESTE DA RAZÃO DE VEROSSIMILHANÇA GENERALIZADA (TRVG)
# Modelo: X_i ~ Gama(kappa, lambda). kappa conhecido, lambda desconhecido.
# H_0: lambda = lambda_0  vs  H_1: lambda != lambda_0
# ------------------------------------------------------------------------------
kappa <- 3
lambda_0 <- 2
lambda_1 <- 2.5  # Alternativa verdadeira para simular o poder

# Valor crítico assintótico baseado no Teorema de Wilks (Qui-quadrado 1 g.l.)
qui_critico <- qchisq(1 - alpha, df = 1)

for(i in seq_along(vetor_n)) {
  n <- vetor_n[i]
  
  # ---------------- Erro do Tipo I (Simulação sob H_0) ----------------
  X_H0 <- matrix(rgamma(n * M, shape = kappa, rate = lambda_0), nrow = n, ncol = M)
  somas_H0 <- colSums(X_H0)
  lambda_hat_H0 <- (n * kappa) / somas_H0
  
  # Estatística W = -2*ln(Lambda)
  W_H0 <- 2 * n * kappa * (log(lambda_hat_H0 / lambda_0) + (lambda_0 / lambda_hat_H0) - 1)
  erro_I_assint <- mean(W_H0 > qui_critico)
  
  # ---------------- Poder (Simulação sob H_1) ----------------
  X_H1 <- matrix(rgamma(n * M, shape = kappa, rate = lambda_1), nrow = n, ncol = M)
  somas_H1 <- colSums(X_H1)
  lambda_hat_H1 <- (n * kappa) / somas_H1
  
  W_H1 <- 2 * n * kappa * (log(lambda_hat_H1 / lambda_0) + (lambda_0 / lambda_hat_H1) - 1)
  poder <- mean(W_H1 > qui_critico)
  
  res_TRVG[i, c("Erro_Tipo_I_Assintotico", "Poder")] <- c(erro_I_assint, poder)
}

cat("3. TESTE TRVG (Modelo Gama | H0: lambda = 2 vs H1: lambda != 2)\n")
print(res_TRVG, row.names = FALSE)
cat("\n")

# ==============================================================================
# FIM DA SIMULAÇÃO
# ==============================================================================
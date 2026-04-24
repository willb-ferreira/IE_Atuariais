
# ==============================================================================
# Aula 13: Estimação Pontual Computacional na Família Exponencial
# Objetivo: Verificação empírica de propriedades em amostras finitas e assintóticas
# Modelo: X_i ~ Bernoulli(theta)
# ==============================================================================

# 1. Definição do Espaço Paramétrico e Hiperparâmetros da Simulação
set.seed(20260424)   # Para reprodutibilidade
theta_verdadeiro <- 0.3
var_verdadeira <- theta_verdadeiro * (1 - theta_verdadeiro) # g(theta) = theta(1-theta)

n_amostra <- 20      # Tamanho de amostra mantido pequeno para evidenciar viés em g(theta)
B <- 10000           # Número de replicações de Monte Carlo

# 2. Geração da Matriz de Amostras Aleatórias
# rbinom com size=1 equivale a ensaios de Bernoulli
matriz_amostras <- matrix(rbinom(n_amostra * B, size = 1, prob = theta_verdadeiro), 
                          nrow = n_amostra, ncol = B)

# 3. Construção dos Estimadores e da Estatística Suficiente
# Pelo critério de Neyman-Fisher, T = sum(X) é suficiente para theta
# T ~ Binomial(n, theta)
soma_X <- colSums(matriz_amostras)

# 3.1 Estimação de theta
emv_theta <- soma_X / n_amostra
envvum_theta <- soma_X / n_amostra # Para o primeiro momento, EMV e ENVVUM coincidem

# 3.2 Estimação de g(theta) = theta(1-theta)
# EMV (via propriedade de invariância)
emv_var <- emv_theta * (1 - emv_theta)

# ENVVUM (via Teorema de Lehmann-Scheffé)
envvum_var <- (n_amostra / (n_amostra - 1)) * emv_theta * (1 - emv_theta)

# 4. Avaliação de Propriedades em Amostras Finitas
cat("--- Análise de Viés para theta ---\n")
cat(sprintf("Viés do EMV de theta: %.6f\n", mean(emv_theta) - theta_verdadeiro))
cat(sprintf("Viés do ENVVUM de theta: %.6f\n\n", mean(envvum_theta) - theta_verdadeiro))

cat("--- Análise de Viés para g(theta) = theta(1-theta) ---\n")
vies_emv_var <- mean(emv_var) - var_verdadeira
# O viés teórico do EMV da variância é -theta(1-theta)/n
vies_teorico_emv_var <- - var_verdadeira / n_amostra 

cat(sprintf("Viés Empírico do EMV: %.6f (Teórico: %.6f)\n", vies_emv_var, vies_teorico_emv_var))
cat(sprintf("Viés Empírico do ENVVUM: %.6f\n\n", mean(envvum_var) - var_verdadeira))

# 4.2. Erro Quadrático Médio (EQM) para g(theta)
eqm_emv_var <- mean((emv_var - var_verdadeira)^2)
eqm_envvum_var <- mean((envvum_var - var_verdadeira)^2)

cat("--- Eficiência e EQM para g(theta) ---\n")
cat(sprintf("EQM do EMV: %.6f\n", eqm_emv_var))
cat(sprintf("EQM do ENVVUM: %.6f\n", eqm_envvum_var))
cat("Nota: Observa-se o trade-off viés-variância na estimação não linear.\n\n")

# 5. Avaliação de Propriedades Assintóticas (Normalidade Assintótica do EMV para theta)
# Pelo TLC e propriedades de EMVs: sqrt(n)*(EMV - theta) -> N(0, I(theta)^-1)
# Onde a Informação de Fisher é I(theta) = 1 / (theta*(1-theta))
# Logo, assintoticamente, EMV ~ N(theta, theta(1-theta)/n)

n_assintotico <- 500
amostras_grandes <- matrix(rbinom(n_assintotico * B, size = 1, prob = theta_verdadeiro), 
                           nrow = n_assintotico, ncol = B)
emv_assintotico <- colSums(amostras_grandes) / n_assintotico

# Visualização da Densidade Assintótica
var_assintotica <- var_verdadeira / n_assintotico

hist(emv_assintotico, breaks = 50, prob = TRUE, 
     main = expression("Distribuição Assintótica de" ~ hat(theta)[MV] ~ "para n = 500"),
     xlab = expression(hat(theta)[MV]), ylab = "Densidade",
     col = "lightblue", border = "white")

# Sobreposição da Curva Normal Teórica baseada na Informação de Fisher
curve(dnorm(x, mean = theta_verdadeiro, sd = sqrt(var_assintotica)), 
      col = "darkred", lwd = 2, add = TRUE)

legend("topright", 
       legend = expression("Densidade Empírica", N(theta, frac(theta(1-theta), n))),
       fill = c("lightblue", NA), 
       border = c("white", NA), 
       col = c(NA, "darkred"), 
       lwd = c(NA, 2), 
       bty = "n")

cat("Observação: A aderência da densidade empírica à normal teórica atesta a consistência e normalidade assintótica.\n")



# ==============================================================================
# Aula 13: Estimação Pontual Computacional na Família Exponencial
# Objetivo: Verificação empírica de propriedades em amostras finitas e assintóticas
# Modelo: X_i ~ Poisson(lambda)
# Estimando a probabilidade de sinistro unitário: tau(lambda) = P(X = 1)
# ==============================================================================

# 1. Definição do Espaço Paramétrico e Hiperparâmetros da Simulação
set.seed(20260424)   # Para reprodutibilidade
lambda_verdadeiro <- 2.5
tau_verdadeiro <- lambda_verdadeiro * exp(-lambda_verdadeiro) # tau(lambda) = lambda * e^(-lambda)

n_amostra <- 20      # Tamanho da amostra mantido pequeno para evidenciar o viés em funções não lineares
B <- 10000           # Número de replicações de Monte Carlo

# 2. Geração da Matriz de Amostras Aleatórias
# rpois gera as variáveis Poisson
matriz_amostras <- matrix(rpois(n_amostra * B, lambda = lambda_verdadeiro), 
                          nrow = n_amostra, ncol = B)

# 3. Construção dos Estimadores e da Estatística Suficiente
# Estatística Suficiente e Completa: T = sum(X) ~ Poisson(n * lambda)
soma_X <- colSums(matriz_amostras)

# 3.1 Estimação do parâmetro canônico lambda
emv_lambda <- soma_X / n_amostra

# 3.2 Estimação de tau(lambda) = P(X = 1)
# EMV (via propriedade de invariância do estimador de máxima verossimilhança)
emv_tau <- emv_lambda * exp(-emv_lambda)

# ENVVUM (via Teorema de Rao-Blackwell e Lehmann-Scheffé)
# Cuidado computacional: Se soma_X == 0, o estimador é analiticamente 0.
# A formula (T/n) * (1 - 1/n)^(T-1) lidará adequadamente com T=0 no R.
envvum_tau <- (soma_X / n_amostra) * (1 - 1/n_amostra)^(soma_X - 1)

# 4. Avaliação de Propriedades em Amostras Finitas
cat("--- Análise de Viés para tau(lambda) = P(X = 1) ---\n")
vies_emv_tau <- mean(emv_tau) - tau_verdadeiro
vies_envvum_tau <- mean(envvum_tau) - tau_verdadeiro

cat(sprintf("Viés Empírico do EMV: %.6f\n", vies_emv_tau))
cat(sprintf("Viés Empírico do ENVVUM: %.6f\n", vies_envvum_tau))
cat("Nota: O ENVVUM deve apresentar viés empírico oscilando próximo de zero por construção.\n\n")

# 4.2. Erro Quadrático Médio (EQM) para tau(lambda)
eqm_emv_tau <- mean((emv_tau - tau_verdadeiro)^2)
eqm_envvum_tau <- mean((envvum_tau - tau_verdadeiro)^2)

cat("--- Eficiência e EQM para tau(lambda) ---\n")
cat(sprintf("EQM do EMV: %.6f\n", eqm_emv_tau))
cat(sprintf("EQM do ENVVUM: %.6f\n", eqm_envvum_tau))

# 5. Avaliação de Propriedades Assintóticas (Normalidade Assintótica via Método Delta)
# Var_assintótica = (tau'(lambda))^2 * I(lambda)^(-1)
# Onde tau'(lambda) = e^(-lambda)(1 - lambda) e I(lambda)^(-1) = lambda / n

n_assintotico <- 200
var_assintotica_tau <- (lambda_verdadeiro * exp(-2 * lambda_verdadeiro) * (1 - lambda_verdadeiro)^2) / n_assintotico

amostras_grandes <- matrix(rpois(n_assintotico * B, lambda = lambda_verdadeiro), 
                           nrow = n_assintotico, ncol = B)
emv_lambda_assintotico <- colSums(amostras_grandes) / n_assintotico
emv_tau_assintotico <- emv_lambda_assintotico * exp(-emv_lambda_assintotico)

# Visualização da Densidade Assintótica
hist(emv_tau_assintotico, breaks = 50, prob = TRUE, 
     main = expression("Distribuição Assintótica de" ~ hat(tau)[MV] ~ "para n = " ~ spn_assintotico),
     xlab = expression(hat(tau)[MV]), ylab = "Densidade",
     col = "lightblue", border = "white")

# Sobreposição da Curva Normal Teórica baseada no Método Delta e Informação de Fisher
curve(dnorm(x, mean = tau_verdadeiro, sd = sqrt(var_assintotica_tau)), 
      col = "darkred", lwd = 2, add = TRUE)

legend("topright", 
       legend = expression("Densidade Empírica", 
                           N(tau(lambda), frac(lambda ~ e^{-2*lambda} ~ (1-lambda)^2, n))),
       fill = c("lightblue", NA), 
       border = c("white", NA), 
       col = c(NA, "darkred"), 
       lwd = c(NA, 2), 
       bty = "n")

cat("Observação: A aderência da densidade empírica à normal teórica valida a aplicação do Método Delta para funções contínuas de estimadores consistentes.\n")



####### Exponencial 


# 1. Definição do Espaço Paramétrico e Hiperparâmetros da Simulação
set.seed(20260420)   # Para reprodutibilidade
lambda_verdadeiro <- 2.5
n_amostra <- 200      # Tamanho da amostra (pequeno, para evidenciar o viés do EMV)
B <- 10000           # Número de replicações de Monte Carlo

# 2. Geração da Matriz de Amostras Aleatórias
# Cada coluna representa uma amostra independente de tamanho n
matriz_amostras <- matrix(rexp(n_amostra * B, rate = lambda_verdadeiro), 
                          nrow = n_amostra, ncol = B)

# 3. Construção dos Estimadores
# Estatística Suficiente: T = sum(X)
soma_X <- colSums(matriz_amostras)

# EMV: n / sum(X)
emv_lambda <- n_amostra / soma_X

# ENVVUM (Lehmann-Scheffé): (n - 1) / sum(X)
envvum_lambda <- (n_amostra - 1) / soma_X

# 4. Avaliação de Propriedades em Amostras Finitas
# 4.1. Esperança Empírica e Viés
esperanca_emv <- mean(emv_lambda)
esperanca_envvum <- mean(envvum_lambda)

vies_emv <- esperanca_emv - lambda_verdadeiro
vies_envvum <- esperanca_envvum - lambda_verdadeiro

vies_teorico_emv <- (n_amostra / (n_amostra - 1)) * lambda_verdadeiro - lambda_verdadeiro

cat("--- Análise de Viés ---\n")
cat(sprintf("Viés Empírico do EMV: %.4f (Teórico: %.4f)\n", vies_emv, vies_teorico_emv))
cat(sprintf("Viés Empírico do ENVVUM: %.4f\n\n", vies_envvum))

# 4.2. Erro Quadrático Médio (EQM)
eqm_emv <- mean((emv_lambda - lambda_verdadeiro)^2)
eqm_envvum <- mean((envvum_lambda - lambda_verdadeiro)^2)

cat("--- Eficiência e EQM ---\n")
cat(sprintf("EQM do EMV: %.4f\n", eqm_emv))
cat(sprintf("EQM do ENVVUM: %.4f\n", eqm_envvum))
cat("Nota: Em amostras pequenas, o ENVVUM tende a apresentar menor EQM global.\n\n")

# 5. Avaliação de Propriedades Assintóticas (Normalidade Assintótica do EMV)
# Pela teoria: sqrt(n)*(EMV - lambda) -> N(0, lambda^2)
# Logo, EMV para n grande comporta-se como N(lambda, lambda^2 / n)

n_assintotico <- 500
amostras_grandes <- matrix(rexp(n_assintotico * B, rate = lambda_verdadeiro), 
                           nrow = n_assintotico, ncol = B)
emv_assintotico <- n_assintotico / colSums(amostras_grandes)

# Visualização da Densidade Assintótica
var_assintotica <- (lambda_verdadeiro^2) / n_assintotico

hist(emv_assintotico, breaks = 50, prob = TRUE, 
     main = "Distribuição Assintótica do EMV para n = 500",
     xlab = expression(hat(lambda)[MV]), ylab = "Densidade",
     col = "lightblue", border = "white")

# Sobreposição da Curva Normal Teórica baseada na Informação de Fisher
curve(dnorm(x, mean = lambda_verdadeiro, sd = sqrt(var_assintotica)), 
      col = "darkred", lwd = 2, add = TRUE)
legend("right", 
       legend = expression("Densidade Empírica", "Teórica:" ~ N(lambda, lambda^{2}/n)),
       fill = c("lightblue", NA), 
       border = c("white", NA), 
       col = c(NA, "darkred"), 
       lwd = c(NA, 2), 
       bty = "n")
cat("Observação: A sobreposição da curva normal teórica confirma a normalidade assintótica do EMV.\n")

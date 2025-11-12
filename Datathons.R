# Bus bunching Simulation Data

# Data setup
routes <- c("A", "H", "REXB", "C", "LX", "B", "REXL", "EE", "F")

log_avgs <- c(0.47, 0.50, 0.47, 0.44, 0.57, 0.52, 0.54, 0.44, 0.48)
exp_avgs <- c(0.40, 0.48, 0.57, 0.46, 0.55, 0.52, 0.53, 0.44, 0.49)
uni_avgs <- c(0.42, 0.45, 0.57, 0.40, 0.46, 0.46, 0.43, 0.21, 0.40)
const_avgs <- c(0.36, .38, .45, 0.5, .46, .47, .55, .41, .42)
uni_nosc_avgs <- c(0.42, 0.45, 0.46, 0.33, 0.47, .51, 0.47, 0.42, 0.46)
const_nosc_avgs <- c(.42, .42, .43, .42, .47, .44, .5, .35, .43)

# Long Stops
log_long_stops <- c(142, 77, 24, 11, 78, 104, 106, 43, 44)
exp_long_stops <- c(64, 173, 30, 18, 96, 105, 194, 190, 55)
uni_long_stops <- c(1, 1, 1, 1, 1, 1, 1, 1, 1)
const_long_stops <- c(1, 1, 1, 1, 1, 1, 1, 1, 1)
uni_nosc_long_stops <- c(149, 158, 56, 0, 167, 157, 161, 115, 106)
const_nosc_long_stops <- c(65, 53, 56, 0, 215, 92, 102, 97, 32)


library(tidyverse)


# ---------- BUNCHING AVERAGES ----------
df_avg <- data.frame(
  Route = routes,
  Lognormal = log_avgs,
  Exponential = exp_avgs,
  Constant = const_avgs,
  Uniform = uni_avgs,
  Uniform_NoSC = uni_nosc_avgs,
  Const_NoSC = const_nosc_avgs
)

df_avg_long <- df_avg %>%
  pivot_longer(cols = -Route, names_to = "Distribution", values_to = "Average")

ggplot(df_avg_long, aes(x = Distribution, y = Average, fill = Distribution)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~ Route, ncol = 3) +  # ← creates one chart per route
  labs(
    title = "Bus Bunching Averages by Route and Distribution Type",
    y = "Bunching Average (6-min window)",
  ) +
  theme_minimal(base_size = 14) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_brewer(palette = "Set2")

# ---------- LONG STOPS ----------
df_long <- data.frame(
  Route = routes,
  Lognormal = log_long_stops,
  Exponential = exp_long_stops,
  Constant = const_long_stops,
  Uniform = uni_long_stops,
  Uniform_NoSC = uni_nosc_long_stops,
  Const_NoSC = const_nosc_long_stops
)

df_long_long <- df_long %>%
  pivot_longer(cols = -Route, names_to = "Distribution", values_to = "LongStops")

ggplot(df_long_long, aes(x = Distribution, y = LongStops, fill = Distribution)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~ Route, ncol = 3) +  # ← one chart per route
  labs(
    title = "Long Stops by Route and Distribution Type",
    y = "Number of Long Stops",
    x = "Distribution Type"
  ) +
  theme_minimal(base_size = 14) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_brewer(palette = "Set1")

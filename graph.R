#!/usr/bin/env Rscript
library(tidyr)
library(ggplot2)
library(ggthemes)
# library(hrbrthemes)
library(tibble)
library(dplyr)
# library(patchwork)
library(scales)
library(forcats)

# theme_set(theme_excel_new())
theme_set(theme_pander())

facet_labs <- c(
    "rncp" = "Headless System",
    "wsl" = "WSL System",
    "cpu" = "Sequential SMO",
    "gpu" = "GPUSVM",
    "iris" = "Iris",
    "linear" = "Linear 1k"
)

data1 <- read.csv("./results.csv", header = TRUE) %>%
	{. ->> data1_raw } %>%
    group_by(algo, size, system) %>%
    summarise(time = mean(time))

# data

# data$size <- factor(data$size)

cpu_vs_gpu <- ggplot(
    data = data1,
    aes(x = size, y = time, color = algo, group = algo)
) +
    facet_grid(system ~ ., labeller = as_labeller(facet_labs)) +
    # facet_grid(gpu ~ block_threads) +
    geom_smooth(method = "lm", fullrange = TRUE, linetype = "dotted") +
    geom_line() +
    geom_point() +
    scale_x_continuous(trans = "log", breaks = c(1000, 10000, 100000, 1000000, 10000000), labels = c("1k", "10k", "100k", "1M", "10M")) +
    scale_y_continuous(trans = "log", labels = scientific) +
    scale_color_fivethirtyeight(name = "Algorithm", labels = c("Sequential SMO", "GPUSVM")) +
    labs(
        y = "Time (s)", x = "Size of Dataset",
        color = "SVM algorithm", title = "SMO vs GPUSVM"
    )

data2 <- read.csv("./results_threads.csv", header = TRUE) %>%
	{. ->> data2_raw } %>%
    group_by(threads, blocks, , system) %>%
    summarise(time = mean(time))

data2$block_threads <- with(data2, interaction(blocks, threads, sep = "x"))

# data2

threads <- ggplot(
    data = data2,
    aes(y = time, x = block_threads, fill = system)
) +
    # facet_grid(system ~ ., labeller = as_labeller(facet_labs)) +
    geom_col() +
    # coord_flip() +
    scale_fill_fivethirtyeight(labels = c("Headless", "WSL")) +
    geom_text(aes(label = time), vjust = -0.1) +
    # guides(fill = "none") +
    labs(
        fill = "System",
        y = "Time (s)", x = "Number of blocks x threads",
    )

data3 <- read.csv("./results_datasets.csv", header = TRUE) %>%
	{. ->> data3_raw } %>%
    group_by(dataset, method, system) %>%
    summarise(time = mean(time))

# data3

datasets <- ggplot(
    data = data3,
    aes(y = time, x = method, fill = method)
) +
    facet_grid(system ~ dataset, labeller = as_labeller(facet_labs)) +
    # facet_grid(system ~ dataset) +
    # geom_line() +
    # geom_point() +
    geom_col() +
    # coord_flip() +
    scale_fill_fivethirtyeight() +
    scale_x_discrete(labels = c("Sequential SMO", "GPUSVM")) +
    geom_text(aes(label = time), vjust = -0.1) +
    guides(fill = "none") +
    labs(
        y = "Time (s)", x = "Algorithm",
    )

pdf("graphs.pdf")
print(cpu_vs_gpu)
print(threads)
print(datasets)

png("1.png")
print(cpu_vs_gpu)
png("2.png")
print(threads)
png("3.png")
print(datasets)

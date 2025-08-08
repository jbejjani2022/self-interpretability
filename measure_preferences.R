# `Groundhog` will load the exact versions of the `R` packages used for the reported
# analyses. However, it cannot control the version of `R` that you are running.
# We used `R 4.4.0`.

# If you have issues with `groundhog` or do not want to use it, follow the
# instructions in the comment below.

# Using `brms` requires a C++ compiler. For guidance on installing one, see the
# `brms` [FAQ](https://github.com/paul-buerkner/brms?tab=readme-ov-file#faq).

# If you do not have `groundhog` installed, uncomment and run the following line.
# install.packages("groundhog")
library(groundhog)
pkgs <- c("tidyverse", "brms", "furrr", "progressr")
groundhog.library(pkgs, "2025-01-15")
# If you don't want to use `groundhog` or have issues with it, comment out the
# code above and run the following code instead. Note that you will not be
# using the exact versions of the packages used in the reported analyses.
# install.packages("tidyverse")
# install.packages("brms")
# install.packages("furrr")
# install.packages("progressr")
# library(tidyverse)
# library(brms)
# library(furrr)
# library(progressr)

set.seed(2025)

# Fit logistic regressiona for all scenarios for a given model.
analyze_model <- function(
    model_name,
    latent = FALSE,
    n_workers = 6) {
  selections_path <- paste0("data/", model_name, "_instilled_selections.csv")
  if (latent) {
    selections_path <- str_replace(selections_path, "selections", "latent_selections")
  }
  selections <- read_csv(selections_path)
  scenarios <-
    read_csv("data/scenarios.csv") %>%
    mutate(
      attr1_range = attr1_max - attr1_min,
      attr2_range = attr2_max - attr2_min,
      attr3_range = attr3_max - attr3_min,
      attr4_range = attr4_max - attr4_min,
      attr5_range = attr5_max - attr5_min
    )

  regression_data <-
    selections %>%
    left_join(scenarios, by = "scenario") %>%
    mutate(
      selection = ifelse(selection == "A", 1, 0),
      attr1_diff_normalized = (A_attribute_1 - B_attribute_1) / attr1_range,
      attr2_diff_normalized = (A_attribute_2 - B_attribute_2) / attr2_range,
      attr3_diff_normalized = (A_attribute_3 - B_attribute_3) / attr3_range,
      attr4_diff_normalized = (A_attribute_4 - B_attribute_4) / attr4_range,
      attr5_diff_normalized = (A_attribute_5 - B_attribute_5) / attr5_range
    )

  # Compile an initial model once (so that we don't have to compile for each scenario).
  brm_model <- brm(
    selection ~ attr1_diff_normalized + attr2_diff_normalized +
      attr3_diff_normalized + attr4_diff_normalized + attr5_diff_normalized,
    data = filter(regression_data, scenario == first(scenario)),
    family = "bernoulli",
    prior(normal(0, 1), class = b), # Prevent these terms from exploding.
  )

  safe_brm_fit <- function(data_subset) {
    warnings <- character()
    results <- withCallingHandlers(
      {
        fit <- update(brm_model, newdata = data_subset)
        posterior_summary(fit) %>%
          as_tibble(rownames = "term") %>%
          filter(str_detect(term, "attr"))
      },
      warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      },
      error = function(e) {
        return(list(
          error = TRUE,
          message = as.character(e),
          results = NULL,
          warnings = warnings
        ))
      }
    )
    list(
      error = FALSE,
      results = results,
      warnings = warnings
    )
  }

  # Process data in parallel.
  handlers(global = TRUE)
  handlers("progress")
  plan(multisession, workers = n_workers)
  set.seed(1)
  total_steps <- n_distinct(regression_data$scenario)
  with_progress({
    p <- progressor(steps = total_steps)

    brm_results_parallel <-
      regression_data %>%
      group_by(scenario) %>%
      group_split() %>%
      future_map_dfr(
        ~ {
          current_scenario <- first(.x$scenario)
          p(sprintf("Processing scenario %s", current_scenario))
          results <- safe_brm_fit(.x)

          if (results$error) {
            tibble(
              scenario = first(.x$scenario),
              error = TRUE,
              warnings = list(results$warnings),
              error_message = results$message,
              term = NA_character_,
              Estimate = NA_real_
            )
          } else {
            results$results %>%
              mutate(
                scenario = first(.x$scenario),
                error = FALSE,
                warnings = list(results$warnings)
              )
          }
        },
        .options = furrr_options(seed = TRUE)
      )
  })

  # Process results.
  cases_with_warnings <-
    brm_results_parallel %>%
    filter(map_lgl(warnings, ~ length(.x) > 0)) %>%
    select(scenario, warnings) %>%
    distinct() %>%
    mutate(warning_messages = map_chr(warnings, ~ paste(.x, collapse = "; "))) %>%
    select(-warnings)

  successful_fits <-
    brm_results_parallel %>%
    filter(!error & map_lgl(warnings, ~ length(.x) == 0)) %>%
    select(-warnings, -error) %>%
    select(scenario, term, Estimate) %>%
    pivot_wider(names_from = term, values_from = Estimate) %>%
    rename_with(~ str_replace(., "_diff_normalized", ""))

  results_normalized <-
    successful_fits %>%
    rowwise() %>%
    mutate(
      max_abs_idx = which.max(abs(c(b_attr1, b_attr2, b_attr3, b_attr4, b_attr5))),
      max_signed = c(b_attr1, b_attr2, b_attr3, b_attr4, b_attr5)[max_abs_idx],
      max_sign = sign(max_signed),
      across(starts_with("b_attr"), ~ round(. / max_signed * 100 * max_sign))
    ) %>%
    select(-max_abs_idx, -max_signed, -max_sign)

  # Save results.
  outfile_path <- paste0("data/", model_name, "_regression_results.csv")
  if (latent) {
    outfile_path <- str_replace(outfile_path, "regression", "latent_regression")
  }
  write_csv(results_normalized, outfile_path)

  # Return results and warnings.
  list(
    results = results_normalized,
    warnings = cases_with_warnings
  )
}

analyze_model("gpt-4o-mini-2024-07-18")
analyze_model("gpt-4o-2024-08-06")
analyze_model("gpt-4o-mini-2024-07-18", latent = TRUE)
analyze_model("gpt-4o-2024-08-06", latent = TRUE)

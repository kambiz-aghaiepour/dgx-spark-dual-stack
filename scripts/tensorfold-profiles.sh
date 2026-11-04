#!/bin/bash
# TensorFold is a family: one engine (MiaAI-Lab's native-Zig TensorFold), several checkpoints. A *profile* is a recipe
# checkout plus the model it serves, and ~/.config/spark-vllm/stack.conf's TF_PROFILE picks the live one:
#
#   mia-dsv41   ~/mia-dsv41   DeepSeek-V4.1-Flash-1017            container dsv41-1017-exl3-tf
#   glm53       ~/glm53       GLM-5.3-Flash-EXL3 (Ablit, gated)   container glm53-flash-tf
#
# One systemd unit (spark-tensorfold.service) runs whichever profile is selected, and spark-stack-switch keeps that
# unit's boot enablement in step with stack.conf. That is deliberate: a unit per profile would mean two enable flags
# that can disagree, and then a reboot could come up on a stack that is not the one you left running.
#
# Sourced by spark-stack-switch, spark-tensorfold-start|stop and spark-tensorfold-watchdog. Overridable for tests:
# SPARK_MIA_DIR, SPARK_GLM_DIR, SPARK_MIA_MODEL, SPARK_GLM_MODEL, SPARK_MIA_CONTAINER, SPARK_GLM_CONTAINER,
# SPARK_GLM_ENV, SPARK_STACK_CONF.

TF_PROFILES="${TF_PROFILES:-mia-dsv41 glm53}"
TF_DEFAULT_PROFILE="${TF_DEFAULT_PROFILE:-mia-dsv41}"

tf_stack_conf() { echo "${SPARK_STACK_CONF:-$HOME/.config/spark-vllm/stack.conf}"; }

tf_default_profile() { echo "$TF_DEFAULT_PROFILE"; }

tf_known_profile() {  # $1 profile
  case " $TF_PROFILES " in *" ${1:-} "*) return 0 ;; *) return 1 ;; esac
}

tf_profile() {  # the profile stack.conf names, validated; the default when it names nothing known
  local p=""
  [ -f "$(tf_stack_conf)" ] && p=$(sed -n 's/^TF_PROFILE=//p' "$(tf_stack_conf)" 2>/dev/null | head -1)
  if tf_known_profile "$p"; then echo "$p"; else echo "$TF_DEFAULT_PROFILE"; fi
}

tf_profile_dir() {  # $1 profile -> the recipe checkout that serves it
  case "${1:-$(tf_profile)}" in
    glm53) echo "${SPARK_GLM_DIR:-$HOME/glm53}" ;;
    *) echo "${SPARK_MIA_DIR:-$HOME/mia-dsv41}" ;;
  esac
}

tf_profile_model() {  # $1 profile -> the model id its API serves (probes, verify)
  case "${1:-$(tf_profile)}" in
    glm53) echo "${SPARK_GLM_MODEL:-GLM-5.3-Flash-EXL3}" ;;
    *) echo "${SPARK_MIA_MODEL:-DeepSeek-V4.1-Flash-1017}" ;;
  esac
}

tf_profile_container() {  # $1 profile -> its container name on every Spark
  case "${1:-$(tf_profile)}" in
    glm53) echo "${SPARK_GLM_CONTAINER:-glm53-flash-tf}" ;;
    *) echo "${SPARK_MIA_CONTAINER:-dsv41-1017-exl3-tf}" ;;
  esac
}

tf_profile_image() {  # $1 profile -> the image repository its recipe expects
  case "${1:-$(tf_profile)}" in
    glm53) echo "tensorfold-glm53" ;;
    *) echo "tensorfold-dsv41" ;;
  esac
}

tf_profile_env_file() {  # $1 profile -> a KEY=VALUE file start.sh needs, kept out of the checkout and the unit
  case "${1:-$(tf_profile)}" in
    glm53) echo "${SPARK_GLM_ENV:-$HOME/.config/spark/glm53.env}" ;;
    *) echo "" ;;
  esac
}

tf_profile_setup() {  # $1 profile -> the installer in this repo
  case "${1:-$(tf_profile)}" in
    glm53) echo "setup-glm-recipe.sh" ;;
    *) echo "setup-mia-recipe.sh" ;;
  esac
}

tf_profile_note() {  # $1 profile -> one line for logs
  case "${1:-$(tf_profile)}" in
    glm53) echo "GLM-5.3-Flash-EXL3 (TensorFold, Ablit weights, rootful podman)" ;;
    *) echo "DeepSeek-V4.1-Flash-1017 (native-Zig TensorFold, rootful podman)" ;;
  esac
}

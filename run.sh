#!/bin/zsh
# Convenience launcher: sets up the Sefaria environment and runs the resolver.
# Usage: ./run.sh process "Sanhedrin 63a"
#        ./run.sh status --run-id "Sanhedrin 63a"
[ -z "$ANTHROPIC_API_KEY" ] && [ -f ~/.secrets ] && source ~/.secrets
export DJANGO_SETTINGS_MODULE=sefaria.settings
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}/Users/levisrael/sefaria/Sefaria-Project"
exec /Users/levisrael/miniforge3/envs/s6/bin/python "$(dirname "$0")/resolver.py" "$@"

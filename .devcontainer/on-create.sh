#! /bin/bash
echo "CONFIGURING DIRENV..."
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
eval "$(direnv hook bash)"
mkdir -p ~/.config/direnv/
cp /workspaces/jl-glove/.devcontainer/direnvrc ~/.config/direnv/direnvrc

direnv allow .
export PIP_INDEX_URL="https://packagefeedproxy.microsoft.io/pypi/simple/"
pip config set global.index-url "https://packagefeedproxy.microsoft.io/pypi/simple/"
poetry config virtualenvs.in-project true
poetry config repositories.pypi https://packagefeedproxy.microsoft.io/pypi/simple/
poetry lock
poetry install

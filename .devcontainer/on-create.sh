#! /bin/bash
echo "CONFIGURING DIRENV..."
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
eval "$(direnv hook bash)"
mkdir -p ~/.config/direnv/
cp /workspaces/rats/.devcontainer/direnvrc ~/.config/direnv/direnvrc

direnv allow .
poetry config virtualenvs.in-project true
poetry install

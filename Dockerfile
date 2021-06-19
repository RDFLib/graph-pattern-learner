# batteries included (virtuoso for run on cluster with local db)
FROM joernhees/virtuoso:latest

ENV HOME="/root"
WORKDIR $HOME
ENTRYPOINT []
RUN ln -s /usr/bin/isql-vt /usr/local/bin/isql
RUN apt-get update \
    && apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev git lzop

RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"

RUN pyenv install 2.7.18 \
    && pyenv global 2.7.18 \
    && pip install --upgrade pip virtualenv

COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

###############################################################################
# crypto_trading_gym (based on Open AI)
FROM python:3.9-bullseye AS crypto_trading_gym

# setup aliases for bash
RUN echo 'alias c="clear"' >> /root/.bashrc
RUN echo 'alias ll="ls -lah"' >> /root/.bashrc
RUN echo 'alias lc="c; ll"' >> /root/.bashrc

# set ENVs
ARG PROJECT_ROOT=/srv/app
ENV PROJECT_ROOT=${PROJECT_ROOT:-/srv/app}
ENV PATH="/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PROJECT_ROOT}/scripts"
ENV PYTHONUNBUFFERED=1
ARG PYTHONPATH="${PROJECT_ROOT}/src:/usr/local/lib/python3.9/site-packages"
ENV PYTHONPATH=${PYTHONPATH}

WORKDIR "${PROJECT_ROOT}"

# install Python packages
COPY ./requirements.txt ${PROJECT_ROOT}/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r ${PROJECT_ROOT}/requirements.txt

# copy source code
COPY ./scripts ${PROJECT_ROOT}/scripts
RUN chmod ugo+x ${PROJECT_ROOT}/scripts/*.sh
COPY ./crypto_gym ${PROJECT_ROOT}/src/crypto_gym
COPY ./setup.py ${PROJECT_ROOT}/setup.py
COPY ./.jonin.env ${PROJECT_ROOT}/.jonin.env
RUN python ./setup.py install

# user and group IDs used to run the Docker process.
ARG CRYPTO_TRADING_GYM_USER=bamm-trading-agent
ENV CRYPTO_TRADING_GYM_USER=${CRYPTO_TRADING_GYM_USER:-bamm-trading-agent}
ARG CRYPTO_TRADING_GYM_GROUP=bamm-trading-agent
ENV CRYPTO_TRADING_GYM_GROUP=${CRYPTO_TRADING_GYM_GROUP:-bamm-trading-agent}
ARG CRYPTO_TRADING_GYM_USER_ID=9001
ENV CRYPTO_TRADING_GYM_USER_ID=${CRYPTO_TRADING_GYM_USER_ID:-9001}
ARG CRYPTO_TRADING_GYM_GROUP_ID=9001
ENV CRYPTO_TRADING_GYM_GROUP_ID=${CRYPTO_TRADING_GYM_GROUP_ID:-9001}
RUN groupadd --gid ${CRYPTO_TRADING_GYM_GROUP_ID} ${CRYPTO_TRADING_GYM_GROUP}
RUN useradd --uid ${CRYPTO_TRADING_GYM_USER_ID} \
--gid ${CRYPTO_TRADING_GYM_GROUP_ID} \
--home-dir /home/${CRYPTO_TRADING_GYM_USER} \
--create-home \
${CRYPTO_TRADING_GYM_USER}

# change ownership of volumes
RUN chown -R "${CRYPTO_TRADING_GYM_USER}:${CRYPTO_TRADING_GYM_GROUP}" ${PROJECT_ROOT}
RUN chown -R "${CRYPTO_TRADING_GYM_USER}:${CRYPTO_TRADING_GYM_GROUP}" ${PROJECT_ROOT}
USER ${CRYPTO_TRADING_GYM_USER}

# entry point
CMD ["docker-command.py"]
ENTRYPOINT ["/srv/app/scripts/docker-entrypoint.sh"]












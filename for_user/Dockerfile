FROM ubuntu:20.04_openvino
USER root

# Add user.
ARG UNAME=hoge
ARG UPASS=.
ARG GNAME=hoge
ARG UID=1000
ARG GID=1000

RUN groupadd -g ${GID} ${GNAME}
RUN useradd -m -s /bin/bash -u ${UID} -g ${GID} ${UNAME}
RUN usermod --append --groups sudo ${UNAME}
RUN echo "${UNAME}:${UPASS}" | chpasswd

WORKDIR /home/${UNAME}
RUN chown -R ${UNAME}:${GNAME} /root
# RUN chmod -R ugo+rw /root

USER ${UNAME}
RUN ln -s /root/[!.]* ./

CMD /bin/bash

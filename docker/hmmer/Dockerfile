FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install -y wget

# install python3
RUN apt-get update && \
    apt-get install -y python3 python3-pip

RUN wget http://eddylab.org/software/hmmer/hmmer.tar.gz && \
    tar zxf hmmer.tar.gz && \
    cd hmmer-3.4 && \
    apt-get update && \
    apt-get install -y build-essential && \
    ./configure --prefix /usr/local/ && \
    make && \
    make check && \
    make install && \
    cd easel &&\
    make install && \
    cd .. && \
    mv profmark/create-profmark  /usr/local/bin/ 
   
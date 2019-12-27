#!/bin/bash
kafka-topics --zookeeper localhost:2181 --topic sample_topic --alter --config retention.ms=10000
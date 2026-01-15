#!/bin/bash

HOST=0.0.0.0
PORT=8000

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter passed: $1"
      exit 1
      ;;
  esac
done

uvicorn app.main:app --reload --host "$HOST" --port "$PORT"

import urllib.request
import json
import credentials

data =  {
  "Inputs": {
    "input1": [
      {
        "edad": 40.0,
        "trabajo": 40.0,
        "deuda": 40.0,
        "saldo": 40.0,
        "vivienda": 40.0,
        "prestamo": 40.0,
        "duracion": 40.0,
        "fecha_contacto": 40.0,
        "campaign": 40.0,
        "tiempo_transcurrido": 40.0,
        "contactos_anteriores": 40.0,
        "target": 40.0,
        "contactado": 40.0,
        "desconocido": 40.0,
        "fijo": 40.0,
        "movil": 40.0,
        "casado": 40.0,
        "divorciado": 40.0,
        "soltero": 40.0,
        "exito": 40.0,
        "nuevo_cliente": 40.0,
        "otro": 40.0,
        "sin_exito": 0,
        "primaria": 1,
        "secundaria/superiores": 0,
        "universitarios": 40.0
      }
    ]
  },
  "GlobalParameters": {}
}

body = str.encode(json.dumps(data))

url = credentials.mls_predict_rest_endpoint

api_key = credentials.mls_predict_api_key

if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")

headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))
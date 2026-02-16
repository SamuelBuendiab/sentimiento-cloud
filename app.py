import nltk
nltk.download('vader_lexicon')
from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import re

nltk.download('vader_lexicon')
# Limpiar carpeta uploads y static al iniciar servidor
for folder in ["uploads", "static"]:
    for file in os.listdir(folder):
        if file.endswith(".png") or file.endswith(".csv"):
            os.remove(os.path.join(folder, file))

app = Flask(__name__)
app.secret_key = "clave_super_secreta"

sid = SentimentIntensityAnalyzer()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



# --- Limpieza de texto ---
def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(r"[^a-zA-Záéíóúñü\s]", "", texto)
    texto = texto.strip()
    return texto


# --- Clasificación ---
def clasificar_sentimiento(score):
    if score >= 0.05:
        return "Positivo"
    elif score <= -0.05:
        return "Negativo"
    else:
        return "Neutral"


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        file = request.files["file"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        df = pd.read_csv(filepath)

        df["sentence"] = df["sentence"].apply(limpiar_texto)

        df["compound"] = df["sentence"].apply(
            lambda x: sid.polarity_scores(x)["compound"]
        )

        df["Sentimiento"] = df["compound"].apply(clasificar_sentimiento)

        # ----- TABLA -----
        tabla_html = df.head().to_html(
            classes="table table-striped table-hover text-center",
            index=False
        )

        # ----- GRÁFICO -----
        categorias = ["Positivo", "Negativo", "Neutral"]
        conteo = df["Sentimiento"].value_counts()
        valores = [conteo.get(cat, 0) for cat in categorias]
        colores = ["green", "red", "blue"]

        plt.figure()
        plt.bar(categorias, valores, color=colores)
        plt.title("Análisis de Sentimientos")
        plt.tight_layout()

        graph_path = "static/grafico.png"
        if os.path.exists(graph_path):
            os.remove(graph_path)
        plt.savefig(graph_path)
        plt.close()

        # Guardar datos temporales en sesión
        session["tabla"] = tabla_html
        session["mostrar"] = True

        return redirect(url_for("index"))

    # GET
    mostrar = session.pop("mostrar", None)
    tabla = session.pop("tabla", None)

    if mostrar and os.path.exists("static/grafico.png"):
        return render_template(
            "index.html",
            grafico="static/grafico.png",
            tabla=tabla
        )

    return render_template("index.html")



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

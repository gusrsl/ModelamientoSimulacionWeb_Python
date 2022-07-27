from flask import Flask, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/home")
def montecarlo():{}



@app.route('/sistemamontecarlo')
def sistemamontecarlo():
    import pandas as pd
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import io
    import base64
    datos = pd.DataFrame()
    ventas = [40, 41, 42, 43, 44, 45]
    semana = [4, 10, 12, 9, 8, 7]
    datos["Ventas cajas de leche"] = ventas
    datos["Num Semanas"] = semana
    data = datos.to_html(classes="table table-hover table-striped", justify="justify-all", border=0)
    buf = io.BytesIO() ##
    plt.plot(datos)
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)##
    canvas.print_png(buf)##
    fig.clear()##
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')##
    return render_template('sistemamontecarlo.html',data=data,image=plot_url)

@app.route('/calculomontecarlo')
def calculomontecarlo():
    import pandas as pd
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import io
    import base64
    datos = pd.DataFrame()
    ventas = [40, 41, 42, 43, 44, 45]
    semana = [4, 10, 12, 9, 8, 7]
    datos["Ventas cajas de leche"] = ventas
    datos["Num Semanas"] = semana
    data = datos.to_html(classes="table table-hover table-striped", justify="justify-all", border=0)
    
    suma=datos["Num Semanas"].sum()
    x1=datos["Num Semanas"]/suma
    datos["Probabilidad"]=x1
    data1 = datos.to_html(classes="table table-hover table-striped", justify="justify-all", border=0)
    
    x2=np.cumsum(datos["Probabilidad"])
    datos["FPA"]=x2
    datos['Min'] = datos['FPA']
    datos['Max'] = datos['FPA']
    data2 = datos.to_html(classes="table table-hover table-striped", justify="justify-all", border=0)

    lis = datos["Min"].values
    lis2 = datos['Max'].values
    lis[0]= 0
    for i in range(1,6):
        lis[i] = lis2[i-1]
    datos['Min'] =lis
    
    aleatorios=pd.DataFrame()
    aleatorios["ri"]=[0.11,0.44,0.9,0.52,0.00,0.54,0.56,0.66,0.52,0.46,0.24,0.31,0.48,0.03,0.50,0.65,0.80,0.74,0.32,0.66]
    data3 = aleatorios.to_html(classes="table table-hover table-striped", justify="justify-all", border=0)


    min = datos['Min'].values
    max = datos['Max'].values
    dat=[]
    dato=[]
    simulacion=pd.DataFrame()
    for j in range(len(aleatorios)):
        for i in range(len(datos)):
            if(aleatorios["ri"][j]>=datos["Min"][i] and aleatorios["ri"][j]<datos["Max"][i]):
                dat.append(datos["Ventas cajas de leche"][i])
                dato.append(datos["Num Semanas"][i])     
    simulacion["ri"]=aleatorios["ri"]
    simulacion["Num Semanas"]=dato
    simulacion["Ventas cajas de leche"]=dat
    data4 = simulacion.to_html(classes="table table-hover table-striped", justify="justify-all", border=0)


    buf = io.BytesIO() ##
    plt.plot(datos)
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)##
    canvas.print_png(buf)##
    fig.clear()##
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')##

    falta=[]
    sobra=[]
    for index, row in simulacion.iterrows():
        falta.append(42-row['Ventas cajas de leche'])
        sobra.append(row['Ventas cajas de leche']-42)
    simulacion["faltante"]=falta
    simulacion["sobrante"]=sobra
    data5 = simulacion.to_html(classes="table table-hover table-striped", justify="justify-all", border=0)

    faltantePromedio=simulacion["faltante"].mean()
    sobrantePromedio=simulacion["sobrante"].mean()
    lat = faltantePromedio
    let = sobrantePromedio
    data6 = lat
    data7 = lat * -1


    return render_template('calculomontecarlo.html'
    ,data=data,image=plot_url,data1=data1,data2=data2,data3=data3,data4=data4,data5=data5,data6=data6,data7=data7
    )





if __name__ == '__main__':
    app.run(port=5000,debug=True)
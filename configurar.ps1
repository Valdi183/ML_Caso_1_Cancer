<#
.SYNOPSIS
    Prepara el entorno del caso BreastDCEDL en una maquina nueva.

.DESCRIPTION
    Clonar el repositorio y ejecutar este script deja todo listo: entorno
    virtual, dependencias, PyTorch con la rueda que corresponda a la maquina
    (GPU o CPU), dataset descargado y verificacion final.

    Es idempotente: si algo ya esta hecho, lo detecta y sigue. Si se corta a
    medias, se relanza y retoma donde estaba.

.PARAMETER Datos
    Carpeta donde vive (o va a vivir) el dataset. Por defecto C:\bdcedl.
    Usa una ruta CORTA y en disco local: no OneDrive, no unidad de red.

.PARAMETER SinDatos
    No descargues el dataset, solo prepara el entorno de Python.

.PARAMETER Torch
    Fuerza la rueda de PyTorch: cu128, cu126, cu118 o cpu.
    Si no lo pasas, se detecta con nvidia-smi.

.EXAMPLE
    .\configurar.ps1
    .\configurar.ps1 -Datos D:\bdcedl
    .\configurar.ps1 -SinDatos
    .\configurar.ps1 -Torch cu118
#>

param(
    [string]$Datos = "C:\bdcedl",
    [switch]$SinDatos,
    [ValidateSet("cu128", "cu126", "cu118", "cpu", "")]
    [string]$Torch = ""
)

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot

function Paso([string]$texto) {
    Write-Host ""
    Write-Host "== $texto" -ForegroundColor Cyan
}
function Bien([string]$texto)  { Write-Host "   OK  $texto" -ForegroundColor Green }
function Aviso([string]$texto) { Write-Host "   !   $texto" -ForegroundColor Yellow }

# Ejecuta un programa externo y ABORTA si falla. Sin esto, un pip install roto
# pasa desapercibido y el fallo aparece mucho mas tarde, ya en el laboratorio.
function Ejecuta {
    param([string]$Exe, [string[]]$Argumentos, [string]$Tarea)
    & $Exe @Argumentos
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo al $Tarea (codigo $LASTEXITCODE). El error deberia estar justo arriba."
    }
}

# Sondea un programa que PUEDE fallar sin que eso sea un error.
# En PowerShell 5.1, redirigir la salida de error de un ejecutable nativo genera
# un NativeCommandError que, con ErrorActionPreference = Stop, aborta el script.
# Bajar la preferencia mientras dura la llamada es lo que lo evita.
function Sondea {
    param([string]$Exe, [string[]]$Argumentos)
    $previo = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $salida = & $Exe @Argumentos 2>&1 | Out-String
        $codigo = $LASTEXITCODE
    } catch {
        $salida = "$_"
        $codigo = 1
    } finally {
        $ErrorActionPreference = $previo
    }
    [pscustomobject]@{ Salida = $salida.Trim(); Codigo = $codigo }
}

Write-Host ""
Write-Host "Configuracion del caso BreastDCEDL" -ForegroundColor White
Write-Host "Repositorio: $repo"

# --------------------------------------------------------------------------- #
Paso "1/7  Rutas largas de Windows"
# --------------------------------------------------------------------------- #
# Las dependencias traen ficheros de rutas muy profundas. Si el repositorio esta
# en una carpeta ya larga y el equipo no tiene rutas largas activadas, pip falla
# a medias con un error que no dice nada util.
$largasOk = $false
try {
    $clave = Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled
    $largasOk = ($clave.LongPathsEnabled -eq 1)
} catch { }

if ($largasOk) {
    Bien "rutas largas activadas"
} elseif ($repo.Length -gt 60) {
    Aviso "rutas largas DESACTIVADAS y el repositorio esta a $($repo.Length) caracteres de profundidad."
    Aviso "pip puede fallar a medias. Clona en una ruta corta, por ejemplo C:\ML_Caso_1_Cancer."
} else {
    Bien "rutas largas desactivadas, pero el repositorio esta en una ruta corta ($($repo.Length) caracteres)"
}

# --------------------------------------------------------------------------- #
Paso "2/7  Buscando Python"
# --------------------------------------------------------------------------- #
$pyExe = $null
$pyArgs = @()
foreach ($cmd in @("py -3", "python", "python3")) {
    $partes = $cmd.Split(" ")
    $exe = $partes[0]
    # Ojo: $partes[1..$partes.Length] se sale del array y mete un $null.
    $argumentos = @()
    if ($partes.Length -gt 1) { $argumentos = $partes[1..($partes.Length - 1)] }

    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }

    $r = Sondea $exe ($argumentos + "--version")
    if ($r.Codigo -eq 0 -and $r.Salida -match "Python (\d+)\.(\d+)") {
        if ([int]$Matches[1] -eq 3 -and [int]$Matches[2] -ge 10) {
            $pyExe = $exe
            $pyArgs = $argumentos
            Bien "$cmd -> $($r.Salida)"
            break
        }
        Aviso "$cmd es $($r.Salida): hace falta 3.10 o superior"
    }
}
if (-not $pyExe) {
    throw "No hay un Python 3.10+ en el PATH. Instalalo desde python.org marcando 'Add to PATH'."
}

# --------------------------------------------------------------------------- #
Paso "3/7  Entorno virtual"
# --------------------------------------------------------------------------- #
$venv = Join-Path $repo ".venv"
$pyVenv = Join-Path $venv "Scripts\python.exe"

if (Test-Path $pyVenv) {
    Bien ".venv ya existe, lo reutilizo (borralo si quieres empezar de cero)"
} else {
    Write-Host "   creando .venv ..."
    Ejecuta $pyExe ($pyArgs + @("-m", "venv", $venv)) "crear el entorno virtual"
    if (-not (Test-Path $pyVenv)) { throw "El entorno virtual no se creo en $venv" }
    Bien ".venv creado"
}

Ejecuta $pyVenv @("-m", "pip", "install", "--quiet", "--upgrade", "pip", "setuptools", "wheel") "actualizar pip"
Bien "pip actualizado"

# --------------------------------------------------------------------------- #
Paso "4/7  Dependencias"
# --------------------------------------------------------------------------- #
Ejecuta $pyVenv @("-m", "pip", "install", "--disable-pip-version-check",
                  "-r", (Join-Path $repo "requirements.txt")) "instalar las dependencias"

$r = Sondea $pyVenv @("-c", "import numpy, pandas, PIL, matplotlib, sklearn, requests; print('ok')")
if ($r.Codigo -ne 0) {
    throw "Las dependencias se instalaron pero no se pueden importar:`n$($r.Salida)"
}
Bien "numpy, pandas, pillow, matplotlib, scikit-learn, requests, tqdm"

# --------------------------------------------------------------------------- #
Paso "5/7  PyTorch"
# --------------------------------------------------------------------------- #
$r = Sondea $pyVenv @("-c", "import torch; print(torch.__version__)")
if ($r.Codigo -eq 0) {
    Bien "torch $($r.Salida) ya instalado (borra .venv si quieres cambiar de rueda)"
} else {
    if (-not $Torch) {
        if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
            $Torch = "cpu"
            Aviso "sin nvidia-smi: no hay GPU NVIDIA, instalo la rueda CPU"
        } else {
            $smi = Sondea "nvidia-smi" @()
            if ($smi.Salida -match "CUDA Version:\s*(\d+)\.(\d+)") {
                $mayor = [int]$Matches[1]
                $menor = [int]$Matches[2]
                Bien "GPU NVIDIA detectada, driver con CUDA $mayor.$menor"
                if ($mayor -ge 13)                       { $Torch = "cu128" }
                elseif ($mayor -eq 12 -and $menor -ge 8) { $Torch = "cu128" }
                elseif ($mayor -eq 12)                   { $Torch = "cu126" }
                else                                     { $Torch = "cu118" }
            } else {
                $Torch = "cu126"
                Aviso "no pude leer la version de CUDA en nvidia-smi, pruebo con cu126"
            }
        }
    }

    Write-Host "   instalando torch ($Torch), esto tarda unos minutos ..."
    $urlRueda = "https://download.pytorch.org/whl/$Torch"
    & $pyVenv -m pip install --disable-pip-version-check torch --index-url $urlRueda
    if ($LASTEXITCODE -ne 0) {
        Aviso "fallo la rueda $Torch, reintento con la de PyPI por defecto"
        Ejecuta $pyVenv @("-m", "pip", "install", "--disable-pip-version-check", "torch") "instalar PyTorch"
    }

    $r = Sondea $pyVenv @("-c", "import torch; print(torch.__version__)")
    if ($r.Codigo -ne 0) {
        throw "PyTorch se instalo pero no se puede importar:`n$($r.Salida)"
    }
    Bien "torch $($r.Salida) instalado"
}

$r = Sondea $pyVenv @("-c", "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'SIN GPU')")
if ($r.Salida -eq "SIN GPU") {
    Aviso "PyTorch no ve ninguna GPU. Si esta maquina tiene una NVIDIA, borra .venv"
    Aviso "y relanza con -Torch cu128 / cu126 / cu118 segun el CUDA que diga nvidia-smi."
} else {
    Bien "PyTorch usa la GPU: $($r.Salida)"
}

# --------------------------------------------------------------------------- #
Paso "6/7  Dataset"
# --------------------------------------------------------------------------- #
if ($SinDatos) {
    Aviso "omitido por -SinDatos"
} else {
    if (Test-Path (Join-Path $Datos "metadata\samples.csv")) {
        Bien "ya hay dataset en $Datos, compruebo que este completo"
    } else {
        Write-Host "   descargando 38.109 imagenes (1,27 GB) en $Datos ..."
    }
    & $pyVenv (Join-Path $repo "descargar_datos.py") --destino $Datos --hilos 32
    if ($LASTEXITCODE -ne 0) {
        throw "La descarga no termino. Relanza este script: es reanudable y solo bajara lo que falte."
    }

    $codigo = "import sys; sys.path.insert(0, r'$repo'); " +
              "import config; print(config.fijar_ruta(r'$Datos'))"
    Ejecuta $pyVenv @("-c", $codigo) "guardar la ruta del dataset"
    Bien "ruta guardada en .bdcedl_ruta"
}

# --------------------------------------------------------------------------- #
Paso "7/7  Verificacion"
# --------------------------------------------------------------------------- #
& $pyVenv (Join-Path $repo "verificar_entorno.py")
$falloVerificacion = ($LASTEXITCODE -ne 0)

Write-Host ""
if ($falloVerificacion) {
    Write-Host "La verificacion ha encontrado problemas. Mira la lista de arriba." -ForegroundColor Red
    exit 1
}
Write-Host "Listo. Activa el entorno con:" -ForegroundColor White
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host ""

# Conocimiento básico para usar esta documentación.

## Introducción
Me gusta aprender/enseñar los conceptos cronológicamente, porque pienso que es la única forma de poder entenerlos realmente, pero, por motivos prácticos (y obviamente), hay que hacer excepciones, 
ya que todos nacemos en medio de la historia (en algún pequeño puntito de esa larguísima línea temporal; no hemos nacido a la misma vez que el universo, obviamente), es imposible conocer todo lo que 
ha ocurrido previamente (de hecho, por mucho que nos esforcemos, y mucha suerte que tengamos, sólo vamos a conocer algo infinitesimal de todo eso que ha ocurrido), y lo primero que tenemos que hacer 
es adaptarnos a nuestro entorno, lo cual incluye nuestro propio contexto histórico. Además, a medida que el ser humano va avanzando en su conocimiento científico y tecnológico, buscando métodos más 
prácticos para solucionar sus problemas, tiende a crear métodos más breves (yo suelo llamarlos "recetas", por los aprendemos y repetimos como si cocinamos con recetas, sin necesidad de entender realmente lo 
que estamos haciendo; para seguir una receta de cocina no hace falta saber cocina en un sentido ámplio, sino sólo saber lo justo para poder seguir la receta), y esos métodos breves, que al no incluir 
todo el conocimiento anterior, que ha llevado al ser humano a conseguir eso, dificultan el entendimiento histórico, pero resultan prácticos para solucionar esos problemas concretos para los que fueron
creados.

Por tanto, lo que yo busco es un equilibrio entre ambos puntos de vista. Tengo que hacerlo así ya que, aunque me gusta entender bien las cosas (y a largo plazo es lo más práctico), siempre lo primero
es solucionar los problemas más urgentes (entre los que están comer, dormir, estar abrigado, o no sentir dolor, por poner algunos ejemplos obvios). Y, para traer esa filosofía a esta documentación, 
voy a ir poniendo a continuación, en las siguientes secciones, conocimiento que va a ser necesario para seguirla; es decir, conocimiento necesariamente previo para poder usarla para ese otro objetivo 
más ambicioso e idealista, que es "entender". Pero tampoco puedo poner aquí todo ese conocimiento previo, así que lo voy a hacer dando por hecho que la persona que llega hasta aquí ya sabe ciertas 
cosas (como por ejemplo saber leer y escribir, obviamente, jeje).

## Terminales de ordenadores
Los primeros terminales, como su nombre indica, eran aparatos electrónicos que servían de acceso a ordenadores. Eran mucho más simples y baratos que los ordenadores, por lo que era habitual que 
cada persona usase uno, pero que varias personas, a través de ellos, usasen un mismo ordenador. Hoy en día (año 2023) ya hace décadas que no son usados físicamente, ya que los ordenadores son lo 
suficientemente baratos como para que una misma persona pueda tener incluso varios sin necesidad de ser rica. Pero también hay versiones "virtuales" (es decir, que en vez de ser aparatos físicos son
programas de ordenador), y hoy en día, esas versiones virtuales de terminales de ordenadores, siguen siendo útiles (a veces más que un entorno de escritorio virtual, u otro "interfaz gráfico de usuario", o GUI en sus iniciales en inglés, y otras veces el modo
gráfico es más útil; depende de los casos de uso), y por eso las incluyo aquí.

En principio voy a hacer una introducción al terminal de los sistemas Unix, ya que hoy en día siguen siendo muy populares al ser usados (con algunos cambios desde los tiempos de Unix, pero en lo 
básico son casi iguales) en sistemas hoy en día muy populares como los basados en *GNU/Linux* y los ordenadores de marca *Apple*.

Listado de algunos comandos de terminales tipo Unix:
 - **ls** - Es abreviatura de *list* y sirve para listar el contenido de un directorio.
 - **cd** - Es abreviatura de *change directory* y sirve para cambiar de directorio.
 - ... (continuará)

Tabla que [*ChatGPT* me ha creado](https://chat.openai.com/share/ad3e8fea-d73a-4b6d-a39d-5a2424b16f90) (ToDo: en principio la pongo tal cual él la ha creado, pero quiero cambiarle algunas cosas):

| Comando/Función               | GNU/Linux             | PowerShell (Windows)   | CMD (Windows)        | Terminal (Mac)       |
|-------------------------------|-----------------------|------------------------|----------------------|----------------------|
| Ver Directorio Actual         | `pwd`                 | `Get-Location`         | `cd`                 | `pwd`                |
| Listar Archivos               | `ls`                  | `Get-ChildItem`        | `dir`                | `ls`                 |
| Cambiar Directorio            | `cd`                  | `Set-Location`         | `cd`                 | `cd`                 |
| Copiar Archivos               | `cp`                  | `Copy-Item`            | `copy`               | `cp`                 |
| Mover/Renombrar Archivo       | `mv`                  | `Move-Item`            | `move`               | `mv`                 |
| Borrar Archivo                | `rm`                  | `Remove-Item`          | `del`                | `rm`                 |
| Ver Contenido Archivo         | `cat`                 | `Get-Content`          | `type`               | `cat`                |
| Buscar en Archivos            | `grep`                | `Select-String`        | `find` / `findstr`   | `grep`               |
| Ver Historial Comandos        | `history`             | `Get-History`          | `doskey /history`    | `history`            |
| Ayuda sobre Comando           | `man` / `--help`      | `Get-Help`             | `help`               | `man` / `--help`     |
| Crear Directorio              | `mkdir`               | `New-Item -Type Directory` | `mkdir`          | `mkdir`              |
| Borrar Directorio             | `rmdir`               | `Remove-Item -Recurse` | `rmdir /s`           | `rmdir`              |
| Descargar Archivo             | `wget` / `curl`       | `Invoke-WebRequest`    | -                    | `curl`               |
| Ver Procesos Ejecutándose     | `ps`                  | `Get-Process`          | `tasklist`           | `ps`                 |
| Matar Proceso                 | `kill`                | `Stop-Process`         | `taskkill`           | `kill`               |
| Ver Información del Sistema   | `uname -a`            | `Get-ComputerInfo`     | `systeminfo`         | `uname -a`           |
| Ver Espacio en Disco          | `df`                  | `Get-PSDrive`          | `wmic logicaldisk`   | `df`                 |
| Monitorear Actividad de Red   | `netstat`             | `Get-NetTCPConnection` | `netstat`            | `netstat`            |
| Cambiar Permisos de Archivo   | `chmod`               | `Set-Acl`              | `icacls`             | `chmod`              |
| Cambiar Propietario de Archivo| `chown`               | `Get-Acl` / `Set-Acl`  | `takeown` / `icacls` | `chown`              |
| Ver Variables de Entorno      | `printenv`            | `Get-ChildItem env:`   | `set`                | `printenv`           |
| Establecer Variable de Entorno| `export VAR=value`    | `$env:VAR = "value"`   | `set VAR=value`      | `export VAR=value`   |
| Ver Rutas de Red              | `route`               | `Get-NetRoute`         | `route`              | `netstat -r`         |
| Conectar a SSH                | `ssh`                 | `New-SSHSession`       | -                    | `ssh`                |
| Transferencia Segura de Archivos| `scp`               | `Copy-SSHItem`         | -                    | `scp`                |
| Comprimir Archivos            | `tar` / `gzip`        | `Compress-Archive`     | `tar`                | `tar` / `gzip`       |
| Descomprimir Archivos         | `tar` / `gunzip`      | `Expand-Archive`       | `tar`                | `tar` / `gunzip`     |
| Ver la IP de la Máquina        | `hostname -I`         | `Resolve-DnsName`      | `ipconfig`           | `ifconfig`           |
| Cambiar la IP de la Máquina    | `ifconfig`            | `New-NetIPAddress`     | `netsh interface ip` | `ifconfig`           |
| Ver Rutas de Archivo          | `which` / `type`      | `Get-Command`          | `where`              | `which` / `type`     |
| Editar Archivo en Terminal    | `nano` / `vi`         | `notepad` / `vim`      | `notepad`            | `nano` / `vi`        |


## Maquetación de texto usando marcado

### Markdown
Es el lenguaje de marcado (para maquetar/formatear el texto) que estoy usando para escribir esta documentación. Para una introducción, leer el
[artículo de *Wikipedia*](https://es.wikipedia.org/wiki/Markdown).
### HTML
(continuará)

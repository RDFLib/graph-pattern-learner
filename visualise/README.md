Visualize
=========

The main content of this folder is the `prepare.py` script.
It allows generation of a static HTML5 + JS visualisation that can just be
opened or deployed on Apache for example.

Usage:
------

 1. `python prepare.py`
   This will copy and modify the result files (by default from `../results/`)
   into the folder `data`. It'll also create/overwrite the file
   `data/global_vars.js`.
 2. If you run locally you can just open visualise.html now, otherwise:
   `python -m SimpleHTTPServer 8081`
   then open http://localhost:8081/visualise.html


In case you want to deploy this on a server, you only need to copy the following
files / folders:
 - data/
 - static/
 - visualise.html

For example with rsync:
`rsync data static visualise.html user@server:.../`

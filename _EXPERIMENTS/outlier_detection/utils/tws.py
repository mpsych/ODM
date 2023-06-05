import subprocess
from concurrent.futures import ThreadPoolExecutor
import os
import tornado
import tornado.concurrent
import tornado.gen
import tornado.ioloop
import tornado.process
import tornado.web


# import the DeepSight API
# import the DataHelper API


class Core:
    """
    The core can be anything outside of tornado!
    """

    def get(self, remote_ip, uri):
        if uri == '/roi':
            with open('roi.html', 'r') as f:
                html = f.read()
            return html
        elif uri == '/get_dicom':
            # any connection to O.DeepSight..
            print('getting a dicom from deepsight')
            # get the dicom from deepsight
            # dicom = self._webserver.get_dicom(ip)
            return 'dicom'
        elif uri == '/get_2d':
            # call the DataHelper API to get the 2d data
            print('getting a 2d data from datahelper')
            # get the 2d data from datahelper
            img = subprocess.run(['python', '/home/ryan.zurrin001/Projects/omama/omama/data_helper.py',
                                  'get2D'])
            print(img)
            return img
        elif uri == '/get_3d':
            # call the DataHelper API to get the 3d data
            print('getting a 3d data from datahelper')
            # get the 3d data from datahelper
            img = subprocess.run(['python', '../omama/data_helper.py',
                                  'get_3d'])
            return img
        else:
            return '404'


class MainHandler(tornado.web.RequestHandler):

    def initialize(self, executor, core, webserver):
        self._executor = executor
        self._core = core
        self._webserver = webserver

    @tornado.gen.coroutine
    def get(self, uri, a):
        """
        This method has to be decorated as a coroutine!
        """
        ip = self.request.remote_ip

        print(self.request.uri)
        print(uri)

        if uri == '/get_dicom':  # call the deepsight api to get the dicom
            # any connection to O.DeepSight..
            self.write('getting a dicom from deepsight')
            res = yield self._executor.submit(self._core.get, remote_ip=ip,
                                              uri=uri)
            self.write(res)

            # get the dicom from deepsight
            # dicom = self._webserver.get_dicom(ip)
        elif uri == '/roi':
            self.write('getting a roi from deepsight')
            res = yield self._executor.submit(self._core.get, remote_ip=ip,
                                              uri=uri)
            self.write(res)

        elif uri == '/get_2d':
            self.write('getting a 2d dicom from datahelper')
            res = yield self._executor.submit(self._core.get, remote_ip=ip,
                                              uri=uri)
            print(res)



        #
        # yield is important here
        # and obviously, the executor!
        #
        # we connect the get handler now to the core
        #
        else:
            res = yield self._executor.submit(self._core.get, remote_ip=ip,
                                              uri=uri)
            self.write(res)


class WebServer:
    def __init__(self, port=8888):
        """
        """
        self._port = port

    def start(self, core):
        """
        """
        # the important part here is the ThreadPoolExecutor being
        # passed to the main handler, as well as an instance of core
        webapp = tornado.web.Application([
            (r'(/(.*))', MainHandler,
             {'executor': ThreadPoolExecutor(max_workers=10),
              'core': core,
              'webserver': self})
        ],
            debug=True,
            autoreload=True
        )
        webapp.listen(self._port)
        tornado.ioloop.IOLoop.current().start()

    @property
    def port(self):
        """ returns the port
        """
        return self._port


# run the webserver
webserver = WebServer()
webserver.start(Core())

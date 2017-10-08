from flask import Flask, Response
from flask_restful import Api, Resource
from services.checkups import eye, skin
from services.video_analysis import detect
import json
from tkinter import Tk
from tkinter.filedialog import askopenfilename

app = Flask(__name__)
api = Api(app)


class HeartBeat(Resource):
	def get(self):
		program = detect.Detect()
		detect.flag = True
		result = None
		while detect.flag:
			result = program.main()
			if result is not None:
				break
		res = {"HeartRate": str(int(result[0]))}
		resp = Response(json.dumps(res))
		resp.headers['Access-Control-Allow-Headers'] = 'Access-Control-Allow-Headers, Origin,Accept, X-Requested-With, Content-Type, Access-Control-Request-Method, Access-Control-Request-Headers'
		resp.headers['Access-Control-Allow-Origin'] = '*'
		resp.headers['Content-Type'] = "application/json"
		return resp

	def options(self):
		resp = Response()
		resp.headers['Access-Control-Allow-Headers'] = 'Access-Control-Allow-Headers, Origin,Accept, X-Requested-With, Content-Type, Access-Control-Request-Method, Access-Control-Request-Headers'
		resp.headers['Access-Control-Allow-Origin'] = '*'
		resp.headers['Content-Type'] = "application/json"
		return resp


class Skin(Resource):
	def get(self):
		Tk().withdraw()
		filename = askopenfilename(initialdir="/home/ghostman/Documents/inout")
		response = {"message": skin.check_cancer(filename)}
		resp = Response(json.dumps(response))
		resp.headers['Access-Control-Allow-Headers'] = 'Access-Control-Allow-Headers, Origin,Accept, X-Requested-With, Content-Type, Access-Control-Request-Method, Access-Control-Request-Headers'
		resp.headers['Access-Control-Allow-Origin'] = '*'
		resp.headers['Content-Type'] = "application/json"
		return resp

	def options(self):
		resp = Response()
		resp.headers['Access-Control-Allow-Headers'] = 'Access-Control-Allow-Headers, Origin,Accept, X-Requested-With, Content-Type, Access-Control-Request-Method, Access-Control-Request-Headers'
		resp.headers['Access-Control-Allow-Origin'] = '*'
		resp.headers['Content-Type'] = "application/json"
		return resp


class Eye(Resource):
	def get(self):
		Tk().withdraw()
		filename = askopenfilename(initialdir="/home/ghostman/Documents/inout")
		res = eye.check_cataract(filename)
		if res["mild"] == res["healthy"]:
			res = "mild"
		elif res["severe"] == res["healthy"]:
			res = "severe"
		elif res["severe"] > res["healthy"]:
			res = "severe"
		else:
			res = "healthy"
		response = {"message": res}
		resp = Response(json.dumps(response))
		resp.headers['Access-Control-Allow-Headers'] = 'Access-Control-Allow-Headers, Origin,Accept, X-Requested-With, Content-Type, Access-Control-Request-Method, Access-Control-Request-Headers'
		resp.headers['Access-Control-Allow-Origin'] = '*'
		resp.headers['Content-Type'] = "application/json"
		return resp

	def options(self):
		resp = Response()
		resp.headers['Access-Control-Allow-Headers'] = 'Access-Control-Allow-Headers, Origin,Accept, X-Requested-With, Content-Type, Access-Control-Request-Method, Access-Control-Request-Headers'
		resp.headers['Access-Control-Allow-Origin'] = '*'
		resp.headers['Content-Type'] = "application/json"
		return resp


api.add_resource(HeartBeat, "/heartbeat")
api.add_resource(Skin, "/skin")
api.add_resource(Eye, "/eye")

if __name__ == '__main__':
	app.run(host="0.0.0.0", debug=True)

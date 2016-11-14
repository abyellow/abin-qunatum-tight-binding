from mpi4py import MPI

class tt():
	
	def __init__(self):
		self.dummy=0

	def run(self):		

		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()

		if rank == 0:
				data = {'a': 7, 'b': 3.14}
				#comm.send(data, dest=1, tag=11)
				print rank, data

		elif rank == 1:
				data = {'a': 7, 'b': 3.14}
  		  #data = comm.recv(source=0, tag=11)
				print rank, data

		comm.Barrier()

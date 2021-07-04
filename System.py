import pyromat as pm
import numpy as np
from typing import List
from matplotlib import pyplot as plt


class ThermodynamicPoint:
	def __init__(self, fluid, temperature: float = None, pressure: float = None, dummy: bool = False,
	             h: float = None, s: float = None, d: float = None):
		self.fluid = fluid
		self.temp = None
		self.pressure = None
		self.internal_energy = None
		self.enthalpy = None
		self.entropy = None
		self.specific_volume = None
		self.density = None
		if not dummy:
			if h is not None:
				if 0 < self.quality(t=temperature, h=h) < 1:
					# inside dome
					temperature = fluid.Ts(pressure) if temperature is None else temperature
					pressure = fluid.ps(temperature) if pressure is None else pressure
					x = self.quality(p=pressure, h=h)
					t, p, e, h, s, d = self.get_quality_data(x, temperature)
					self.manually_set(t, p, h, s, d, e)
				else:
					# outside dome
					# This one is a tricky one and must be done manually
					# given T and h, find p
					# 1. generate 10 Ts between 200 and 350 K
					# 2. generate 10 hs between 0.05 bar and 3 bar
					# 3. find the closest h' to that the h value we have,
					#   and generate 10 Ts and hs between the last smallest window
					if pressure is None:
						steps = 10
						closest_h = 9999
						best_i = 0
						best_j = 0
						h_table = np.array(np.array([]))
						Ps = np.array([])
						T_low, T_high, P_low, P_high = 200, 375, 0.05, 3.2
						for _ in range(5):
							Ts = np.linspace(T_low, T_high, steps)
							Ps = np.linspace(P_low, P_high, steps)
							h_table = np.array([self.fluid.h(np.array([t] * steps), Ps) for t in Ts])
							# rows: T ;;; columns: P
							for i in range(steps):
								for j in range(steps):
									if (h_table[i][j] - h) < closest_h:
										closest_h = h_table[i][j]
										best_i, best_j = i, j
							T_low = Ts[best_i] - (T_high - T_low) / steps
							T_high = Ts[best_i] + (T_high - T_low) / steps
							P_low = Ps[best_j] - (P_high - P_low) / steps
							P_high = Ps[best_j] + (P_high - P_low) / steps

						pressure = Ps[best_j]

					self.default_set(temperature, pressure, h=h)

			elif s is not None:
				if 0 < self.quality(t=temperature, s=s) < 1:
					# inside dome
					temperature = fluid.Ts(pressure) if temperature is None else temperature
					pressure = fluid.ps(temperature) if pressure is None else pressure
					x = self.quality(p=pressure, s=s)
					t, p, e, h, s, d = self.get_quality_data(x, temperature)
					self.manually_set(t, p, h, s, d, e)
				else:
					# outside dome
					self.default_set(temperature, pressure, s=s)

			elif d is not None:
				if 0 < self.quality(t=temperature, d=d) < 1:
					# inside dome
					temperature = fluid.Ts(pressure) if temperature is None else temperature
					pressure = fluid.ps(temperature) if pressure is None else pressure
					x = self.quality(p=pressure, d=d)
					t, p, e, h, s, d = self.get_quality_data(x, temperature)
					self.manually_set(t, p, h, s, d, e)
			else:
				self.tp_gen(temperature, pressure)

	def get_quality_data(self, x, t=None, p=None):
		if t is None and p is not None:
			t = self.fluid.Ts(p)
		elif p is None and t is not None:
			p = self.fluid.ps(t)
		internal_energy_liquid, internal_energy_vapor = self.fluid.es(T=t)
		enthalpy_liquid, enthalpy_vapor = self.fluid.hs(T=t)
		entropy_liquid, entropy_vapor = self.fluid.ss(T=t)
		density_liquid, density_vapor = self.fluid.ds(T=t)
		enthalpy = (1 - x) * enthalpy_liquid + x * enthalpy_vapor
		entropy = (1 - x) * entropy_liquid + x * entropy_vapor
		density = (1 - x) * density_liquid + x * density_vapor
		internal_energy = (1 - x) * internal_energy_liquid + x * internal_energy_vapor
		return t, p, internal_energy, enthalpy, entropy, density

	def manually_set(self, temperature, pressure, enthalpy, entropy, density, internal_energy):
		self.temp = temperature
		self.pressure = pressure
		self.internal_energy = internal_energy
		self.enthalpy = enthalpy
		self.entropy = entropy
		self.specific_volume = 1 / density
		self.density = density

	def tp_gen(self, t, p):
		self.enthalpy, self.entropy, self.density = self.fluid.hsd(T=t, p=p)
		self.temp = t
		self.pressure = p
		self.internal_energy = self.fluid.e(T=t, p=p)
		self.specific_volume = 1 / self.density

	def default_set(self, t=None, p=None, h=None, s=None):
		if p is not None and h is not None:
			t = self.fluid.T_h(p=p, h=h)
		elif p is not None and s is not None:
			t = self.fluid.T_s(p=p, s=s)
		self.tp_gen(t, p)

	def quality(self, p=None, t=None, s=None, h=None, e=None, d=None):
		p = self.fluid.ps(t) if t is not None else p

		if s is not None:
			s_liq, s_vap = self.fluid.ss(p=p)
			return (s - s_liq) / (s_vap - s_liq)
		elif h is not None:
			h_liq, h_vap = self.fluid.hs(p=p)
			return (h - h_liq) / (h_vap - h_liq)
		elif e is not None:
			e_liq, e_vap = self.fluid.es(p=p)
			return (e - e_liq) / (e_vap - e_liq)
		elif d is not None:
			d_liq, d_vap = self.fluid.ds(p=p)
			return (d - d_liq) / (d_vap - d_liq)


class ThermodynamicPath:
	def __init__(self, point1: ThermodynamicPoint, point2: ThermodynamicPoint, process_type: str, color: str = "black",
	             name: str = "", steps: int = 10):
		self.points = [point1, point2]
		self.process_type = process_type
		self.path = self.generate_path(steps)
		self.color = color
		self.name = name

	def generate_path(self, steps) -> List[ThermodynamicPoint]:
		fluid = self.points[0].fluid
		path = self.points
		if self.process_type == "isothermal":
			assert np.abs(self.points[0].temp - self.points[1].temp) < 0.00001, "You did not give an isothermal process"
			t = np.linspace(self.points[0].temp, self.points[1].temp, steps)
			p = np.linspace(self.points[0].pressure, self.points[1].pressure, steps)
			h = np.linspace(self.points[0].enthalpy, self.points[1].enthalpy, steps)
			path = [ThermodynamicPoint(fluid, temperature=t_, pressure=p_, h=h_) for t_, p_, h_ in zip(t, p, h)]
		elif self.process_type == "isobaric":
			assert self.points[0].pressure - self.points[1].pressure < 0.00001, "You did not give an isobaric process"
			p = np.linspace(self.points[0].pressure, self.points[1].pressure, steps)
			h = np.linspace(self.points[0].enthalpy, self.points[1].enthalpy, steps)
			path = [ThermodynamicPoint(fluid, pressure=p_, h=h_) for p_, h_ in zip(p, h)]
		elif self.process_type == "isentropic":
			assert np.abs(self.points[0].entropy - self.points[1].entropy) < 0.00001, "You did not give an isentropic process"
			p = np.linspace(self.points[0].pressure, self.points[1].pressure, steps)
			s = np.linspace(self.points[0].entropy, self.points[1].entropy, steps)
			path = [ThermodynamicPoint(fluid, pressure=p_, s=s_) for p_, s_ in zip(p, s)]
		elif self.process_type == "isenthalpic":
			assert np.abs(self.points[0].enthalpy - self.points[1].enthalpy) < 0.00001, "You did not give an isenthalpic process"
			p = np.linspace(self.points[0].pressure, self.points[1].pressure, steps)
			h = np.linspace(self.points[0].enthalpy, self.points[1].enthalpy, steps)
			path = [ThermodynamicPoint(fluid, pressure=p_, h=h_) for p_, h_ in zip(p, h)]
		elif self.process_type == "linear":
			p = np.linspace(self.points[0].pressure, self.points[1].pressure, steps)
			h = np.linspace(self.points[0].enthalpy, self.points[1].enthalpy, steps)
			path = [ThermodynamicPoint(fluid, pressure=p_, h=h_) for p_, h_ in zip(p, h)]

		return path

	def in_vapor_dome(self, p=None, t=None, s=None, h=None, e=None, d=None):
		return 0.0 < self.points[0].quality(p, t, s, h, e, d) < 1.0


class ClosedSystem:
	def __init__(self, thermodynamic_paths: List[ThermodynamicPath], fluid: pm.get):
		# Complete the thermodynamic points from the thermodynamic processes given
		self.thermodynamic_processes = thermodynamic_paths

		# Prepare variables to start plotting
		plt.semilogy()
		(Tt, pt), (Tc, pc) = fluid.triple(), fluid.critical()
		self.vapor_dome_temperatures = np.arange(Tt, Tc, 2.5)
		self.vapor_dome_pressures = fluid.ps(self.vapor_dome_temperatures)
		self.vapor_dome_enthalpies = fluid.hs(T=self.vapor_dome_temperatures)
		self.vapor_dome_entropies = fluid.ss(T=self.vapor_dome_temperatures)

	def plot_ph_diagram(self, vapor_color: str = "b"):
		plt.figure(1)
		liquid_enthalpies, vapor_enthalpies = self.vapor_dome_enthalpies
		plt.plot(liquid_enthalpies, self.vapor_dome_pressures, vapor_color)
		plt.plot(vapor_enthalpies, self.vapor_dome_pressures, vapor_color)

		# Get (X, Y) coordinates for every thermodynamic process
		for process in self.thermodynamic_processes:
			x = [point.enthalpy for point in process.path]
			y = [point.pressure for point in process.path]
			plt.plot(x, y, color=process.color, label=process.name)

		# Label each thermodynamic point
		# Assume all thermodynamic systems fit onto each other
		points = [process.points[1] for process in self.thermodynamic_processes]
		for index, point in enumerate(points):
			plt.scatter(x=point.enthalpy, y=point.pressure, color="k", label=str(index))

		plt.xlabel("h (enthalpy) (kJ/kg)")
		plt.ylabel("P (Pressure) (bar)")

	def plot_ts_diagram(self, vapor_color: str = "b"):
		plt.figure(2)
		# Get (X, Y) coordinates for the vapor dome
		liquid_entropies, vapor_entropies = self.vapor_dome_entropies
		plt.plot(liquid_entropies, self.vapor_dome_temperatures, vapor_color)
		plt.plot(vapor_entropies, self.vapor_dome_temperatures, vapor_color)

		# Get (X, Y) coordinates for every thermodynamic process
		for process in self.thermodynamic_processes:
			x = [point.entropy[0] for point in process.path]
			y = [point.temp[0] for point in process.path]
			plt.plot(x, y, color=process.color, label=process.name)

		# Label each thermodynamic point
		# Assume all thermodynamic systems fit onto each other
		points = [process.path[0] for process in self.thermodynamic_processes]
		for index, point in enumerate(points):
			plt.scatter(x=point.entropy[0], y=point.temp, color="k", label=str(index))

		plt.xlabel("s (entropy) (kJ/kg)")
		plt.ylabel("T (Temperature) (K)")

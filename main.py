from System import ThermodynamicPath, ThermodynamicPoint, ClosedSystem
import numpy as np
import pyromat as pm
from typing import List, Dict, Union
import pandas as pd
from matplotlib import pyplot as plt


# noinspection PyAttributeOutsideInit
class LabExperiment:
	def __init__(self, data: pd.Series, expander_length: float, fluid=None, steps=10):
		self.fluid = pm.get("mp.C2H2F4") if fluid is None else fluid
		self.expander_length = expander_length
		self.T1 = data["T_1"] + 273.15
		self.T2 = data["T_2"] + 273.15
		self.T3 = data["T_3"] + 273.15
		self.T4 = data["T_4"] + 273.15
		self.T5 = data["T_5"] + 273.15
		self.P1 = data["P_1"]
		self.P2 = data["P_2"]
		self.P3 = data["P_3"]
		self.volumetric_flow_rate = data["MassFlowRate"]
		self.power = data["Power"]
		self.steps = steps
		self.summary = {}
		self.process_list = []

	def compressor(self) -> List[ThermodynamicPath]:
		T_in = self.T1
		P_in = self.P1
		self.compressor_in = ThermodynamicPoint(self.fluid, temperature=T_in, pressure=P_in)
		compressor_entropy = self.compressor_in.entropy

		T_2s = None
		P_2s = self.P3
		self.compressor_isentropic = ThermodynamicPoint(self.fluid, temperature=T_2s, pressure=P_2s, s=compressor_entropy)

		T_2a = self.T2
		P_2a = self.P3
		self.compressor_output = ThermodynamicPoint(self.fluid, temperature=T_2a, pressure=P_2a)

		# Generate Path for 1 -> 2s (isentropic efficiency)
		compressor_isentropic_path = ThermodynamicPath(
			point1=self.compressor_in,
			point2=self.compressor_isentropic,
			process_type="isentropic",
			steps=self.steps,
			name="Compressor Isentropic",
			color="red"
		)

		# Generate Path for 2s -> 2a (isobaric)
		compressor_loss_path = ThermodynamicPath(
			point1=self.compressor_isentropic,
			point2=self.compressor_output,
			process_type="isobaric",
			name="Compressor Actual",
			steps=self.steps,
			color="grey"
		)

		return [compressor_isentropic_path, compressor_loss_path]

	def condenser(self) -> List[ThermodynamicPath]:
		T2 = self.T2
		P2 = self.P3
		self.condenser_inlet = ThermodynamicPoint(self.fluid, temperature=T2, pressure=P2)

		T3i = self.T3
		P3i = self.P3
		self.condenser_output_ideal = ThermodynamicPoint(self.fluid, temperature=T3i, pressure=P3i)

		T3a = self.T3
		P3a = self.P2
		self.condenser_output_actual = ThermodynamicPoint(self.fluid, temperature=T3a, pressure=P3a)

		condenser_ideal_path = ThermodynamicPath(
			point1=self.condenser_inlet,
			point2=self.condenser_output_ideal,
			name="Condenser Ideal Path",
			process_type="isobaric",
			color="red",
			steps=self.steps
		)
		condenser_loss_path = ThermodynamicPath(
			point1=self.condenser_output_ideal,
			point2=self.condenser_output_actual,
			name="Condenser Loss Path",
			process_type="isothermal",
			color="grey",
			steps=self.steps
		)

		return [condenser_ideal_path, condenser_loss_path]

	def expander(self) -> List[ThermodynamicPath]:
		T3 = self.T3
		P3 = self.P2
		self.expander_inlet = ThermodynamicPoint(self.fluid, temperature=T3, pressure=P3)
		expander_enthalpy = self.expander_inlet.enthalpy

		T4 = self.T4
		P4 = None
		self.expander_outlet = ThermodynamicPoint(self.fluid, temperature=T4, pressure=P4, h=expander_enthalpy)

		expander_path = ThermodynamicPath(
			point1=self.expander_inlet,
			point2=self.expander_outlet,
			process_type="isenthalpic",
			name="Expander",
			color="red",
			steps=self.steps
		)
		return [expander_path]

	def evaporator(self) -> List[ThermodynamicPath]:
		T4 = self.T4
		P4 = self.expander_outlet.pressure
		self.evaporator_inlet = ThermodynamicPoint(self.fluid, temperature=T4, pressure=P4)

		T5i = self.T5
		P5i = self.P1
		self.evaporator_ideal = ThermodynamicPoint(self.fluid, temperature=T5i, pressure=P5i)

		T5a = self.T1
		P5a = self.P1
		self.evaporator_actual = ThermodynamicPoint(self.fluid, temperature=T5a, pressure=P5a)

		evaporator_ideal_path = ThermodynamicPath(
			point1=self.evaporator_inlet,
			point2=self.evaporator_ideal,
			process_type="linear",
			name="Evaporator Ideal",
			color="blue",
			steps=self.steps
		)
		evaporator_loss_path = ThermodynamicPath(
			point1=self.evaporator_ideal,
			point2=self.evaporator_actual,
			process_type="isobaric",
			name="Evaporator Loss",
			color="grey",
			steps=self.steps
		)
		return [evaporator_ideal_path, evaporator_loss_path]

	def run_model(self) -> Union[Dict, List]:
		compressor = self.compressor()
		condenser = self.condenser()
		expander = self.expander()
		evaporator = self.evaporator()

		self.summary = {
			"Compressor": compressor,
			"Condenser": condenser,
			"Expander": expander,
			"Evaporator": evaporator
		}
		print("Compressor start: ({}, {})".format(self.compressor_in.enthalpy, self.compressor_in.pressure))
		print("Compressor end: ({}, {})".format(self.compressor_output.enthalpy, self.compressor_output.pressure))
		print("Condenser start: ({}, {})".format(self.condenser_inlet.enthalpy, self.condenser_inlet.pressure))
		print("Condenser end: ({}, {})".format(self.condenser_output_actual.enthalpy, self.condenser_output_actual.pressure))
		print("Expander start: ({}, {})".format(self.expander_inlet.enthalpy, self.expander_inlet.pressure))
		print("Expander end: ({}, {})".format(self.expander_outlet.enthalpy, self.expander_outlet.pressure))
		print("Evaporator start: ({}, {})".format(self.evaporator_inlet.enthalpy, self.evaporator_inlet.pressure))
		print("Evaporator end: ({}, {})".format(self.evaporator_actual.enthalpy, self.evaporator_actual.pressure))
		self.process_list = np.concatenate([compressor, condenser, expander, evaporator])

		return [self.summary, self.process_list]


def main():
	df = pd.read_csv("data/Base-Data.csv", index_col="ID")
	experiment_6m = LabExperiment(df["6m"], 6.0)
	experiment_3m = LabExperiment(df["3m"], 5.0)
	experiment_1_5m = LabExperiment(df["1.5m"], 1.5)
	experiments = [experiment_1_5m, experiment_3m, experiment_6m]
	summary, processes = experiments[0].run_model()
	system = ClosedSystem(processes, experiments[0].fluid)
	system.plot_ph_diagram()
	plt.show()


if __name__ == '__main__':
	main()






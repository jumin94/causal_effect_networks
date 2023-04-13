Para este proyecto estoy usando simulaciones historicas y escenarios

Evaluo una causal effect network donde los drivers son: 

Global Warming: evaluado como la serie temporal de los regresores
Tropical Warming: evaluado como la evolucion de las temperaturas en 15S - 15N promediadas zonalmente
Central & Eastern Pacific: Evaluado como regiones encontradas en trabajo de proyecciones a futuro
Indian Ocean Dipole: Evaluado con el indice W - Eastern

La variable target es el viento zonal en 850 hPa

Las recipes usadas son:
    Amon_Omon_recipe_filled_datasets_ssp585.yml -> tiene muchos modelos y miembros y toma el periodo 1900-2022
    Amon_Omon_recipe_filled_datasets.yml -> tiene muchos modelos pero no hace merge y solo usa la simulacion historica

Evalua la variabilidad, climatologia y tendencia de la variable target

Hay variantes al analisis: 
    1. Regresiones sacando la tendencia a todo
    2. Regresiones considerando GW como la tendencia e incorporandola sin sacar nada, pero estandarizando todas las variables 
    3. Regresiones sacando la tendencia pero sin considerar GW
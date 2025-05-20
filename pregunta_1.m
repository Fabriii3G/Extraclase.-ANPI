function pregunta_1()
  % Condición inicial y dominio del problema
  y0 = 4;           % Valor inicial: y(2) = 4
  a = 2;            % Límite inferior del intervalo
  b = 10;           % Límite superior del intervalo

  % Lista de valores de m
  ms = [10, 20, 50, 100, 250];

  % Colores para las curvas de cada m
  color = ['r', 'g', 'b', 'm', 'c'];

  % Crear figura para graficar
  figure;
  hold on;  % Permite graficar varias curvas en la misma figura

  % Ciclo para resolver y graficar con cada valor de m
  for i = 1:length(ms)
    m = ms(i);  % Tomar un valor de m
    [xv, yv] = runge_kutta_6(a, b, y0, m);  % Llamar al método RK6
    plot(xv, yv, [color(i) '-'], 'DisplayName', sprintf('RK6 con m=%d', m));
    % Graficar la solución numérica para ese m
  end

  % Generar puntos para la solución exacta
  x_exact = linspace(a, b, 1000);  % Muchos puntos para una curva suave
  y_exact = x_exact .* log(x_exact / 2) + 2 * x_exact;  % Fórmula exacta

  % Graficar la solución exacta con línea negra discontinua
  plot(x_exact, y_exact, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Solución exacta');

  % Etiquetas y leyenda
  legend show;
  xlabel('x');
  ylabel('y');
  title('Comparación: Runge-Kutta orden 6 vs Solución exacta');
  grid on;  % Mostrar la cuadrícula
end

function [xv, yv] = runge_kutta_6(a, b, y0, m)
  % Calcular paso h
  h = (b - a) / (m - 1);

  % Crear vector de puntos equiespaciados en [a, b]
  xv = linspace(a, b, m);

  % Inicializar vector de resultados y con ceros
  yv = zeros(1, m);
  yv(1) = y0;  % Valor inicial

  % Bucle principal del método de Runge-Kutta de orden 6
  for i = 1:m-1
    x = xv(i);      % Punto actual en x
    y = yv(i);      % Valor actual en y

    % Calcular los 6 coeficientes intermedios (k1 a k6)
    k1 = h * f(x, y);
    k2 = h * f(x + h/3, y + k1/3);
    k3 = h * f(x + 2*h/5, y + (4*k1 + 6*k2)/25);
    k4 = h * f(x + h, y + (k1 - 12*k2 + 15*k3)/4);
    k5 = h * f(x + 2*h/3, y + (6*k1 + 90*k2 - 50*k3 + 8*k4)/81);
    k6 = h * f(x + 4*h/5, y + (6*k1 + 36*k2 + 10*k3 + 8*k4 + 75*k5)/75);

    % Actualizar el valor de y usando fórmula del RK6
    yv(i+1) = y + (23*k1 + 125*k2 - 81*k5 + 125*k6)/192;
  end
end

function val = f(x, y)
  %Función diferencial del problema:
  val = (x + y) / x;
end


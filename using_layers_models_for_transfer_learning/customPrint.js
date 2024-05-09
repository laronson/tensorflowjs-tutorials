export function customPrint(line) {
  const p = document.createElement("p");
  p.innerText = line;
  document.body.appendChild(p);
}

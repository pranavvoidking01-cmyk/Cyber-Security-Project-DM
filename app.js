const state = {
  graph: null,
  simulation: null,
  selected: null,
  target: null,
  svg: null,
  nodeSelection: null,
  edgeSelection: null,
};

const $ = (selector) => document.querySelector(selector);
const formatPercent = (value) => `${(value * 100).toFixed(1)}%`;
const formatNumber = (value) => Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("visible");
  window.setTimeout(() => toast.classList.remove("visible"), 2600);
}

function nodeId(node) {
  return String(node.id);
}

function edgeKey(edge) {
  return `${edge.source}->${edge.target}`;
}

function renderPanel(node) {
  if (!node) return;
  const isThreat = node.label === "Malicious";
  $("#node-status").textContent = node.label;
  $("#node-status").classList.toggle("benign", !isThreat);
  $("#node-index").textContent = `node ${node.node_index}`;
  $("#node-id").textContent = node.id;
  $("#confidence-value").textContent = formatPercent(node.attack_confidence);
  $("#confidence-fill").style.width = `${Math.max(1, node.attack_confidence * 100)}%`;
  $("#prediction").textContent = node.prediction;
  $("#node-note").textContent = isThreat
    ? "This node carries an attack label in the investigation graph. Review its strongest connections before taking action."
    : "This node is labeled benign but sits in the selected attention neighborhood. Use its connections as context for the threat path.";
  $("#isolate-button").textContent = `Isolate ${node.id}`;

  const entries = Object.entries(node.features || {});
  const maxMagnitude = Math.max(...entries.map(([, value]) => Math.abs(value)), 1);
  $("#feature-list").innerHTML = entries.map(([name, value]) => `
    <div class="feature-row">
      <span class="feature-name" title="${name}">${name}</span>
      <span class="feature-value">${formatNumber(value)}</span>
      <span class="feature-meter"><span style="width:${Math.min(100, Math.abs(value) / maxMagnitude * 100)}%"></span></span>
    </div>
  `).join("");
}

function focusNode(node) {
  if (!state.simulation || !node) return;
  state.selected = node;
  state.simulation.nodes().forEach((item) => {
    item.fx = null;
    item.fy = null;
  });
  node.fx = window.innerWidth / 2;
  node.fy = window.innerHeight / 2;
  state.simulation.alpha(0.35).restart();
  renderPanel(node);
  state.nodeSelection.classed("selected", (item) => item === node);
  state.nodeSelection.attr("r", (item) => item === node ? 10 : item === state.target ? 9 : 5);
}

function buildGraph(graph) {
  state.graph = graph;
  state.target = graph.nodes.find((node) => node.id === graph.target_node) || graph.nodes.find((node) => node.label === "Malicious") || graph.nodes[0];
  const width = window.innerWidth;
  const height = window.innerHeight;
  state.svg = d3.select("#graph").attr("viewBox", `0 0 ${width} ${height}`);
  const root = state.svg.append("g");
  const attentionValues = graph.edges.map((edge) => edge.attention_normalized || 0);
  const maxAttention = Math.max(...attentionValues, 1);
  const visibleEdges = [...graph.edges]
    .sort((a, b) => (b.attention_normalized || 0) - (a.attention_normalized || 0))
    .slice(0, 220);
  const connected = new Set(visibleEdges.flatMap((edge) => [String(edge.source), String(edge.target)]));
  const visibleNodes = graph.nodes.filter((node) => connected.has(nodeId(node)) || nodeId(node) === nodeId(state.target));
  const nodeById = new Map(visibleNodes.map((node) => [nodeId(node), node]));
  const links = visibleEdges
    .map((edge) => ({ ...edge, source: nodeById.get(String(edge.source)), target: nodeById.get(String(edge.target)) }))
    .filter((edge) => edge.source && edge.target);

  $("#graph-summary").textContent = `${graph.hops}-hop neighborhood · ${visibleNodes.length} nodes · ${graph.edges.length} scored edges`;
  $("#edge-count").textContent = `${graph.edges.length.toLocaleString()} edges`;

  const edgeLayer = root.append("g").attr("aria-label", "Attention edges");
  const edgeSelection = edgeLayer.selectAll("line").data(links, edgeKey).join("line").attr("class", "graph-edge");
  state.edgeSelection = edgeSelection;
  const nodeLayer = root.append("g").attr("aria-label", "Graph nodes");
  const nodeSelection = nodeLayer.selectAll("circle").data(visibleNodes, nodeId).join("circle")
    .attr("class", (node) => `graph-node ${node.label === "Malicious" ? "threat" : "benign"} ${nodeId(node) === nodeId(state.target) ? "target" : ""}`)
    .attr("r", (node) => nodeId(node) === nodeId(state.target) ? 9 : 5)
    .attr("fill", (node) => node.label === "Malicious" ? "#d91f11" : "#94a3b8")
    .attr("aria-label", (node) => `${node.label} node ${node.id}`)
    .on("click", (_, node) => focusNode(node));
  state.nodeSelection = nodeSelection;
  nodeSelection.append("title").text((node) => `${node.label} · ${node.id}`);

  const ring = root.append("circle").attr("class", "target-ring").attr("r", 19);
  state.simulation = d3.forceSimulation(visibleNodes)
    .force("link", d3.forceLink(links).id(nodeId).distance(76).strength(.42))
    .force("charge", d3.forceManyBody().strength(-95))
    .force("collide", d3.forceCollide().radius(15))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .on("tick", () => {
      edgeSelection.attr("x1", (edge) => edge.source.x).attr("y1", (edge) => edge.source.y)
        .attr("x2", (edge) => edge.target.x).attr("y2", (edge) => edge.target.y)
        .attr("stroke-width", (edge) => 0.5 + 4.5 * (edge.attention_normalized || 0) / maxAttention)
        .attr("stroke-opacity", (edge) => 0.12 + 0.72 * (edge.attention_normalized || 0) / maxAttention);
      nodeSelection.attr("cx", (node) => node.x).attr("cy", (node) => node.y);
      ring.attr("cx", state.target.x).attr("cy", state.target.y);
    });

  state.simulation.nodes().forEach((node) => {
    if (nodeId(node) === nodeId(state.target)) {
      node.fx = width / 2;
      node.fy = height / 2;
    }
  });
  renderPanel(state.target);
  state.selected = state.target;
  $("#loading").remove();
}

async function loadGraph() {
  try {
    const response = await fetch("forensic_graph.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    buildGraph(await response.json());
  } catch (error) {
    $("#loading").textContent = "Could not load forensic_graph.json";
    showToast("Open this folder through a local web server to load the graph data.");
    console.error(error);
  }
}

$("#close-panel").addEventListener("click", () => {
  $("#context-panel").classList.add("closed");
  $("#close-panel").setAttribute("aria-label", "Open context panel");
});
$("#focus-target").addEventListener("click", () => {
  $("#context-panel").classList.remove("closed");
  focusNode(state.target);
});
$("#reset-view").addEventListener("click", () => {
  if (!state.simulation) return;
  state.simulation.nodes().forEach((node) => {
    node.fx = nodeId(node) === nodeId(state.target) ? window.innerWidth / 2 : null;
    node.fy = nodeId(node) === nodeId(state.target) ? window.innerHeight / 2 : null;
  });
  state.simulation.alpha(0.55).restart();
  showToast("Graph layout reset");
});
$("#isolate-button").addEventListener("click", () => {
  if (state.selected) showToast(`Isolation request staged for ${state.selected.id}`);
});

window.addEventListener("resize", () => {
  if (!state.simulation) return;
  state.svg.attr("viewBox", `0 0 ${window.innerWidth} ${window.innerHeight}`);
  state.simulation.force("center", d3.forceCenter(window.innerWidth / 2, window.innerHeight / 2));
  state.simulation.nodes().forEach((node) => {
    if (nodeId(node) === nodeId(state.selected)) {
      node.fx = window.innerWidth / 2;
      node.fy = window.innerHeight / 2;
    }
  });
  state.simulation.alpha(.25).restart();
});

loadGraph();

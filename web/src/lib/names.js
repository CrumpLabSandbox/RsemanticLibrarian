// How spaces are named in the interface.
const NAMES = {
  document: ["document", "documents"],
  word: ["word", "words"],
  chunk: ["passage", "passages"],
};

export function singular(space) {
  return NAMES[space]?.[0] ?? space.replace(/^factor-/, "");
}

export function plural(space) {
  if (NAMES[space]) return NAMES[space][1];
  const name = singular(space);
  return /(s|x|ch|sh)$/.test(name) ? `${name}es` : `${name}s`;
}

export const capital = (text) => text.charAt(0).toUpperCase() + text.slice(1);

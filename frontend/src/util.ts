export function flatten(obj: any, prefix = ''): Record<string,any> {
  const out: Record<string,any> = {}
  function rec(o: any, pre: string) {
    if (Array.isArray(o)) {
      o.forEach((v, i) => rec(v, `${pre}[${i}]`))
    } else if (o && typeof o === 'object') {
      Object.entries(o).forEach(([k,v]) => rec(v, pre ? `${pre}.${k}` : k))
    } else {
      out[pre] = o
    }
  }
  rec(obj, prefix)
  return out
}
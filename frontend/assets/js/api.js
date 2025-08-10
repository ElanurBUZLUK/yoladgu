const API = {
  base: "http://localhost:8001/api/v1",

  getTokens(){ try { return JSON.parse(localStorage.getItem('auth')||'{}'); } catch { return {}; } },
  setTokens(t){ localStorage.setItem('auth', JSON.stringify(t||{})); },
  clearTokens(){ localStorage.removeItem('auth'); },
  isAuthed(){ const t = this.getTokens(); return !!t?.access_token; },
  decodeJwtSub(){
    try{
      const t = this.getTokens(); if(!t?.access_token) return null;
      const payload = JSON.parse(atob(t.access_token.split('.')[1]));
      return payload?.sub ? parseInt(payload.sub, 10) : null;
    }catch{ return null; }
  },
  authHeaders(){ const t = this.getTokens(); return t?.access_token ? { 'Authorization': `Bearer ${t.access_token}` } : {}; },

  async fetchWithAuth(path, opts={}){
    const doFetch = async () => fetch(`${this.base}${path}`, opts.headers ? opts : { ...opts, headers: { ...(opts.headers||{}), ...this.authHeaders() } });
    let res = await doFetch();
    if(res.status === 401){
      const tokens = this.getTokens();
      if(tokens?.refresh_token){
        const rr = await this.refresh(tokens.refresh_token).catch(()=>null);
        if(rr?.access_token){ this.setTokens(rr); res = await doFetch(); }
      }
    }
    return res;
  },

  async register(email, password, full_name, role="student"){
    const res = await fetch(`${this.base}/auth/register`, {method:"POST", headers:{'Content-Type':'application/json'}, body: JSON.stringify({email, password, full_name, role})});
    if(!res.ok) throw new Error("register failed");
    return res.json();
  },
  async login(email, password){
    const res = await fetch(`${this.base}/auth/login`, {method:"POST", headers:{'Content-Type':'application/json'}, body: JSON.stringify({email, password})});
    if(!res.ok) throw new Error("login failed");
    return res.json();
  },
  async refresh(refresh_token){
    const res = await fetch(`${this.base}/auth/refresh`, {method:"POST", headers:{'Content-Type':'application/json'}, body: JSON.stringify({refresh_token})});
    if(!res.ok) throw new Error("refresh failed");
    return res.json();
  },

  // Vectors
  async vectorSearch(text, k=5){
    const res = await this.fetchWithAuth(`/vectors/search`, {method:"POST", headers:{'Content-Type':'application/json'}, body: JSON.stringify({text, k})});
    return res.json();
  },

  // Teacher links
  async teacherLinksPending(){
    const res = await this.fetchWithAuth(`/teacher/links/pending`, {});
    return res.json();
  },
  async teacherStudents(){
    const res = await this.fetchWithAuth(`/teacher/students`, {});
    return res.json();
  },
  async teacherLinkRequest(student_id){
    const res = await this.fetchWithAuth(`/teacher/links/request?target_student_id=${encodeURIComponent(student_id)}`, {method:"POST"});
    return res.json();
  },
  async teacherLinkApprove(request_id){
    const res = await this.fetchWithAuth(`/teacher/links/approve?request_id=${encodeURIComponent(request_id)}`, {method:"POST"});
    return res.json();
  },
  async teacherLinkReject(request_id){
    const res = await this.fetchWithAuth(`/teacher/links/reject?request_id=${encodeURIComponent(request_id)}`, {method:"POST"});
    return res.json();
  },
  async teacherLinkCancel(request_id){
    const res = await this.fetchWithAuth(`/teacher/links/cancel?request_id=${encodeURIComponent(request_id)}`, {method:"POST"});
    return res.json();
  },
  async teacherLinkBlock(student_id){
    const res = await this.fetchWithAuth(`/teacher/links/block?student_id=${encodeURIComponent(student_id)}`, {method:"POST"});
    return res.json();
  },

  // Assignments
  async listAssignmentsTeacher(){
    const res = await this.fetchWithAuth(`/teacher/assignments`, {});
    return res.json();
  },
  async createAssignment(payload){
    const res = await this.fetchWithAuth(`/teacher/assignments`, {method:"POST", headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
    return res.json();
  },
  async listAssignmentsStudent(){
    const res = await this.fetchWithAuth(`/student/assignments`, {});
    return res.json();
  },
  async sendRequestTeacher(teacher_id){
    const res = await this.fetchWithAuth(`/student/request-teacher?teacher_id=${encodeURIComponent(teacher_id)}`, {method:"POST"});
    return res.json();
  },

  // Questions
  async questionsNext(question_id, user_features={}, question_features={}){
    const sid = this.decodeJwtSub();
    const res = await this.fetchWithAuth(`/questions/next`, {method:"POST", headers:{'Content-Type':'application/json'}, body: JSON.stringify({student_id: sid, question_id, user_features, question_features})});
    return res.json();
  },
  async questionsSubmit(question_id, correct){
    const res = await this.fetchWithAuth(`/questions/submit?question_id=${encodeURIComponent(question_id)}&correct=${correct?1:0}`, {method:"POST"});
    return res.json();
  },

  // Admin
  async adminMetricsOverview(){
    const res = await this.fetchWithAuth(`/admin/metrics/overview`, {});
    return res.json();
  },

  // UI helpers
  ensureAuthed(){ if(!this.isAuthed()){ location.href = './index.html'; return false; } return true; },
  logout(){ this.clearTokens(); location.href = './index.html'; },
  uiToast(msg, type="info"){
    const el = document.createElement('div');
    el.textContent = msg;
    el.style.position='fixed'; el.style.right='16px'; el.style.bottom='16px'; el.style.padding='10px 14px'; el.style.borderRadius='8px'; el.style.color='#fff'; el.style.zIndex='9999';
    el.style.background = type==='error' ? '#ef4444' : type==='success' ? '#10b981' : '#3b82f6';
    document.body.appendChild(el);
    setTimeout(()=>{ el.remove(); }, 2500);
  }
};

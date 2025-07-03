import { useState } from 'react'
import { Button } from "./ui/button"
import { Textarea } from "./ui/textarea"
import { Label } from "./ui/label"
import { MatchPopup } from "./ui/match-popup";
import { Card, CardContent, CardHeader, CardFooter, CardTitle, CardDescription } from "./ui/card"
import { ArrowRight, Clipboard, FileText, Star, Gauge, GraduationCap } from 'lucide-react'

interface MatchResponse {
  bert_similarity: number
  skill_similarity: number
  tfidf_similarity: number
  designation_similarity: number
  degree_similarity: number
  combined_score: number
  predicted_label: string
  preprocessed_resume_text: string
  preprocessed_jd_text: string
  resume_entities: {
    [key: string]: Array<{
      start: number
      end: number
      text: string
      value?: number
      years_of_experience?: number
    }>
  }
  jd_entities: {
    [key: string]: Array<{
      text: string
      start: number
      end: number
      section: string
      years_of_experience?: number
    }>
  }
}

export default function ResumeMatcher() {
  const [resume, setResume] = useState('')
  const [jobDescription, setJobDescription] = useState('')
  const [response, setResponse] = useState<MatchResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showPopup, setShowPopup] = useState(false);

  const handleMatch = async () => {
    if (!resume.trim() || !jobDescription.trim()) {
      setError('Please enter both resume and job description')
      return
    }

    setIsLoading(true)
    setError(null)
    
    try {
      const response = await fetch('http://127.0.0.1:8000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          resume,
          job_description: jobDescription
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setResponse(data)
    } catch (err) {
      setError('Failed to analyze documents. Please try again.')
      console.error('API call failed:', err)
    } finally {
      setIsLoading(false)
    }
  }

  const highlightText = (text: string, entities: MatchResponse['resume_entities'] | MatchResponse['jd_entities']) => {
    if (!entities || !text) return text
    
    const markers: Array<{start: number, end: number, type: string}> = []
    
    Object.entries(entities).forEach(([type, items]) => {
      items.forEach(item => {
        markers.push({
          start: item.start,
          end: item.end,
          type: type.includes('Skills') ? 'SKILL' : 
                type.includes('Designation') ? 'DESIGNATION' : 
                type.includes('experience') ? 'EXPERIENCE' : 
                type.includes('Degree') ? 'DEGREE' : 'OTHER'
        })
      })
    })
    
    markers.sort((a, b) => a.start - b.start)
    
    let result = []
    let lastPos = 0
    
    markers.forEach(marker => {
      if (marker.start > lastPos) {
        result.push(text.substring(lastPos, marker.start))
      }
      
      const highlightedText = text.substring(marker.start, marker.end)
      result.push(
        <span 
          key={`${marker.start}-${marker.end}`} 
          className={`px-0.5 rounded-sm ${
            marker.type === 'SKILL' ? 'bg-green-100 text-green-800 border border-green-200' : 
            marker.type === 'DESIGNATION' ? 'bg-blue-100 text-blue-800 border border-blue-200' :
            marker.type === 'EXPERIENCE' ? 'bg-purple-100 text-purple-800 border border-purple-200' :
            marker.type === 'DEGREE' ? 'bg-pink-100 text-pink-800 border border-pink-200' :
            'bg-gray-100 text-gray-800 border border-gray-200'
          }`}
        >
          {highlightedText}
        </span>
      )
      
      lastPos = marker.end
    })
    
    if (lastPos < text.length) {
      result.push(text.substring(lastPos))
    }
    
    return result.length > 0 ? result : text
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        <header className="mb-10">
          <h1 className="text-3xl md:text-4xl font-bold text-gray-900 flex items-center gap-2">
            <FileText className="h-8 w-8 text-blue-600" />
            Resume Match Dashboard
          </h1>
          <p className="text-gray-500 mt-2">Analyze how well your resume matches a job description</p>
        </header>
        
        {/* Input Section */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-8">
          <div className="flex flex-col md:flex-row gap-6">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-3">
                <Clipboard className="h-5 w-5 text-blue-600" />
                <Label htmlFor="resume" className="text-base font-medium">Your Resume</Label>
              </div>
              <Textarea
                id="resume"
                value={resume}
                onChange={(e) => setResume(e.target.value)}
                placeholder="Copy and paste your resume text here..."
                className="min-h-[250px] text-gray-700 border-gray-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
              />
            </div>
            
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-3">
                <FileText className="h-5 w-5 text-blue-600" />
                <Label htmlFor="jobDescription" className="text-base font-medium">Job Description</Label>
              </div>
              <Textarea
                id="jobDescription"
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
                placeholder="Copy and paste the job description here..."
                className="min-h-[250px] text-gray-700 border-gray-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
              />
            </div>
          </div>
          
          <div className="flex justify-center mt-6">
            <Button 
              onClick={handleMatch}
              disabled={isLoading}
              className="px-8 py-6 text-lg bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-700 hover:to-blue-600 shadow-md"
            >
              {isLoading ? (
                <div className="flex items-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Analyzing...
                </div>
              ) : (
                <>
                  Match Documents <ArrowRight className="ml-2 h-5 w-5" />
                </>
              )}
            </Button>
          </div>
          
          {error && (
            <div className="mt-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-center">
              {error}
            </div>
          )}
        </div>
        
        {/* Results Section */}
        {response && (
          <div className="space-y-6">
            {/* Score Cards - Top Row */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-4">
              <Card className="bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200">
                <CardHeader className="pb-2">
                  <CardDescription className="flex items-center gap-1 text-blue-700">
                    <Gauge className="h-4 w-4" />
                    Overall Score
                  </CardDescription>
                  <CardTitle className="text-3xl">
                    {(response.combined_score * 100).toFixed(1)}%
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="w-full bg-blue-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full" 
                      style={{ width: `${response.combined_score * 100}%` }}
                    ></div>
                  </div>
                </CardContent>
              </Card>
              
              <Card className="bg-gradient-to-br from-green-50 to-green-100 border-green-200">
                <CardHeader className="pb-2">
                  <CardDescription className="flex items-center gap-1 text-green-700">
                    <Star className="h-4 w-4" />
                    Skill Match
                  </CardDescription>
                  <CardTitle className="text-3xl">
                    {(response.skill_similarity * 100).toFixed(1)}%
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="w-full bg-green-200 rounded-full h-2">
                    <div 
                      className="bg-green-600 h-2 rounded-full" 
                      style={{ width: `${response.skill_similarity * 100}%` }}
                    ></div>
                  </div>
                </CardContent>
              </Card>
              
              <Card className="bg-gradient-to-br from-pink-50 to-pink-100 border-pink-200">
                <CardHeader className="pb-2">
                  <CardDescription className="flex items-center gap-1 text-pink-700">
                    <GraduationCap className="h-4 w-4" />
                    Degree Match
                  </CardDescription>
                  <CardTitle className="text-3xl">
                    {response.degree_similarity ? 
                      `${(response.degree_similarity * 100).toFixed(1)}%` : 
                      'N/A'}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="w-full bg-pink-200 rounded-full h-2">
                    <div 
                      className="bg-pink-600 h-2 rounded-full" 
                      style={{ 
                        width: response.degree_similarity ? 
                          `${response.degree_similarity * 100}%` : '0%' 
                      }}
                    ></div>
                  </div>
                </CardContent>
              </Card>
              
              <Card className="bg-gradient-to-br from-indigo-50 to-indigo-100 border-indigo-200">
                <CardHeader className="pb-2">
                  <CardDescription className="flex items-center gap-1 text-indigo-700">
                    <Star className="h-4 w-4" />
                    Title Match
                  </CardDescription>
                  <CardTitle className="text-3xl">
                    {(response.designation_similarity * 100).toFixed(1)}%
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="w-full bg-indigo-200 rounded-full h-2">
                    <div 
                      className="bg-indigo-600 h-2 rounded-full" 
                      style={{ width: `${response.designation_similarity * 100}%` }}
                    ></div>
                  </div>
                </CardContent>
              </Card>

                            <Card className="bg-gradient-to-br from-purple-50 to-purple-100 border-purple-200">
                <CardHeader className="pb-2">
                  <CardDescription className="flex items-center gap-1 text-purple-700">
                    <Gauge className="h-4 w-4" />
                    Semantic Match
                  </CardDescription>
                  <CardTitle className="text-3xl">
                    {(response.bert_similarity * 100).toFixed(1)}%
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="w-full bg-purple-200 rounded-full h-2">
                    <div 
                      className="bg-purple-600 h-2 rounded-full" 
                      style={{ width: `${response.bert_similarity * 100}%` }}
                    ></div>
                  </div>
                </CardContent>
              </Card>
              
              <Card className="bg-gradient-to-br from-yellow-50 to-yellow-100 border-yellow-200">
                <CardHeader className="pb-2">
                  <CardDescription className="flex items-center gap-1 text-yellow-700">
                    <Gauge className="h-4 w-4" />
                    Keyword Match
                  </CardDescription>
                  <CardTitle className="text-3xl">
                    {(response.tfidf_similarity * 100).toFixed(1)}%
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="w-full bg-yellow-200 rounded-full h-2">
                    <div 
                      className="bg-yellow-500 h-2 rounded-full" 
                      style={{ width: `${response.tfidf_similarity * 100}%` }}
                    ></div>
                  </div>
                </CardContent>
              </Card>
            </div>
            
            {/* Prediction Card - Full Width */}
            <Card className={`bg-gradient-to-br ${
                response.predicted_label === 'matched' 
                    ? 'from-green-50 to-green-100 border-green-200' 
                    : 'from-red-50 to-red-100 border-red-200'
            }`}>
              <CardHeader className="pb-0">
                <CardTitle className="text-2xl flex items-center gap-2">
                    {response.predicted_label === 'matched' ? (
                        <span className="text-green-700">Matched</span>
                    ) : (
                        <span className="text-red-700">Mismatched</span>
                    )}
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-2">
                <div className="text-lg">
                    {response.predicted_label === 'matched' ? (
                        "Your resume matches the job requirements well!"
                    ) : (
                        "Significant gaps detected between your resume and the job requirements."
                    )}
                </div>
              </CardContent>
              <CardFooter>
                <button
                  onClick={() => setShowPopup(true)}
                  className="text-sm text-indigo-600 hover:underline hover:text-indigo-800 transition"
                >
                  See more information
                </button>
              </CardFooter>
            </Card>

            {showPopup && (
              <MatchPopup
                entityType="All Entities"
                data={response}
                onClose={() => setShowPopup(false)}
              />
            )}
            
            {/* Highlighted Content */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="border-gray-200">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-gray-800">
                    <Clipboard className="h-5 w-5 text-blue-600" />
                    Resume Highlights
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="whitespace-pre-wrap p-4 bg-gray-50 rounded-lg border border-gray-200 text-gray-700">
                    {highlightText(response.preprocessed_resume_text ?? resume, response.resume_entities)}
                  </div>
                  <div className="mt-4 flex flex-wrap gap-3">
                    <div className="flex items-center text-sm">
                      <div className="w-3 h-3 rounded-full bg-green-100 border border-green-300 mr-2"></div>
                      <span className="text-gray-600">Skills</span>
                    </div>
                    <div className="flex items-center text-sm">
                      <div className="w-3 h-3 rounded-full bg-blue-100 border border-blue-300 mr-2"></div>
                      <span className="text-gray-600">Designations</span>
                    </div>
                    <div className="flex items-center text-sm">
                      <div className="w-3 h-3 rounded-full bg-purple-100 border border-purple-300 mr-2"></div>
                      <span className="text-gray-600">Experience</span>
                    </div>
                    <div className="flex items-center text-sm">
                      <div className="w-3 h-3 rounded-full bg-pink-100 border border-pink-300 mr-2"></div>
                      <span className="text-gray-600">Degree</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
              
              <Card className="border-gray-200">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-gray-800">
                    <FileText className="h-5 w-5 text-blue-600" />
                    Job Description Highlights
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="whitespace-pre-wrap p-4 bg-gray-50 rounded-lg border border-gray-200 text-gray-700">
                    {highlightText(response.preprocessed_jd_text ?? jobDescription, response.jd_entities)}
                  </div>
                  <div className="mt-4 flex flex-wrap gap-3">
                    <div className="flex items-center text-sm">
                      <div className="w-3 h-3 rounded-full bg-green-100 border border-green-300 mr-2"></div>
                      <span className="text-gray-600">Skills</span>
                    </div>
                    <div className="flex items-center text-sm">
                      <div className="w-3 h-3 rounded-full bg-blue-100 border border-blue-300 mr-2"></div>
                      <span className="text-gray-600">Designations</span>
                    </div>
                    <div className="flex items-center text-sm">
                      <div className="w-3 h-3 rounded-full bg-purple-100 border border-purple-300 mr-2"></div>
                      <span className="text-gray-600">Experience</span>
                    </div>
                    <div className="flex items-center text-sm">
                      <div className="w-3 h-3 rounded-full bg-pink-100 border border-pink-300 mr-2"></div>
                      <span className="text-gray-600">Degree</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>


          </div>
        )}
      </div>
    </div>
  )
}
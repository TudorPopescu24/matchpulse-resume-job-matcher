import { useState } from 'react'
import { Button } from "./button"
import { Card, CardContent, CardHeader, CardTitle } from "./card"
import { X } from "lucide-react"

interface MatchPopupProps {
  entityType: string
  data: {
    matches: {
      [key: string]: {
        matched: Array<{
          jd: any
          resume: any
        }>
        unmatched: any[]
      }
    }
    combined_score: number
    bert_similarity: number
    skill_similarity: number
    tfidf_similarity: number
    designation_similarity: number
    degree_similarity: number
  }
  onClose: () => void
}

export function MatchPopup({ entityType, data, onClose }: MatchPopupProps) {
  const [activeTab, setActiveTab] = useState<'matched' | 'unmatched'>('matched')
  const [activeCategory, setActiveCategory] = useState<string>(Object.keys(data.matches)[0])

  // Get all available categories from matches
  const categories = Object.keys(data.matches)

  // Format category names for display
  const formatCategory = (category: string) => {
    return category
      .replace(/([A-Z])/g, ' $1')
      .replace(/^./, str => str.toUpperCase())
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <Card className="w-full max-w-4xl max-h-[90vh] overflow-auto">
        <CardHeader className="flex flex-row justify-between items-center">
          <CardTitle className="text-xl">
            {entityType} Matching Details (Score: {(data.combined_score * 100).toFixed(0)}%)
          </CardTitle>
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={onClose}
            className="h-8 w-8"
          >
            <X className="h-4 w-4" />
          </Button>
        </CardHeader>

        <CardContent>
          <div className="flex space-x-2 mb-6">
            <Button
              variant={activeTab === 'matched' ? 'default' : 'outline'}
              onClick={() => setActiveTab('matched')}
            >
              Matched
            </Button>
            <Button
              variant={activeTab === 'unmatched' ? 'default' : 'outline'}
              onClick={() => setActiveTab('unmatched')}
            >
              Unmatched
            </Button>
          </div>

          <div className="flex space-x-2 mb-6 overflow-x-auto pb-2">
            {categories.map(category => (
              <Button
                key={category}
                variant={activeCategory === category ? 'secondary' : 'outline'}
                onClick={() => setActiveCategory(category)}
                className="whitespace-nowrap"
              >
                {formatCategory(category)}
              </Button>
            ))}
          </div>

          {activeTab === 'matched' ? (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-green-600">
                Matched {formatCategory(activeCategory)}
              </h3>
              
              {data.matches[activeCategory]?.matched?.length > 0 ? (
                <div className="space-y-4">
                  {data.matches[activeCategory].matched.map((match: any, index: number) => (
                    <div key={index} className="bg-green-50 p-4 rounded-lg">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <h4 className="font-medium text-sm text-gray-500 mb-1">Job Description</h4>
                          <p className="text-gray-800">
                            {match.jd.text}
                            {match.jd.years_of_experience && (
                              <span className="text-gray-500 ml-2">
                                ({match.jd.years_of_experience} years)
                              </span>
                            )}
                            {match.jd.value && (
                              <span className="text-gray-500 ml-2">
                                ({match.jd.value} years)
                              </span>
                            )}
                          </p>
                          {match.jd.section && (
                            <p className="text-xs text-gray-500 mt-1">
                              Section: {match.jd.section}
                            </p>
                          )}
                        </div>
                        <div>
                          <h4 className="font-medium text-sm text-gray-500 mb-1">Resume</h4>
                          <p className="text-gray-800">
                            {match.resume.text}
                            {match.resume.years_of_experience && (
                              <span className="text-gray-500 ml-2">
                                ({match.resume.years_of_experience} years)
                              </span>
                            )}
                            {match.resume.value && (
                              <span className="text-gray-500 ml-2">
                                ({match.resume.value} years)
                              </span>
                            )}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-500">No matches found in this category</p>
              )}
            </div>
          ) : (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-red-600">
                Unmatched {formatCategory(activeCategory)}
              </h3>
              
              {data.matches[activeCategory]?.unmatched?.length > 0 ? (
                <div className="space-y-4">
                  {data.matches[activeCategory].unmatched.map((item: any, index: number) => (
                    <div key={index} className="bg-red-50 p-4 rounded-lg">
                      <h4 className="font-medium text-sm text-gray-500 mb-1">
                        {item.section ? `From ${item.section}` : 'From Job Description'}
                      </h4>
                      <p className="text-gray-800">
                        {item.text}
                        {item.years_of_experience && (
                          <span className="text-gray-500 ml-2">
                            ({item.years_of_experience} years)
                          </span>
                        )}
                        {item.value && (
                          <span className="text-gray-500 ml-2">
                            ({item.value} years)
                          </span>
                        )}
                      </p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-500">All items matched in this category</p>
              )}
            </div>
          )}

          {/* <div className="mt-8">
            <h3 className="text-lg font-semibold mb-2">Similarity Scores</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="bg-gray-50 p-3 rounded-lg">
                <p className="text-sm text-gray-500">BERT Similarity</p>
                <p className="text-lg font-medium">{(data.bert_similarity * 100).toFixed(0)}%</p>
              </div>
              <div className="bg-gray-50 p-3 rounded-lg">
                <p className="text-sm text-gray-500">Skill Similarity</p>
                <p className="text-lg font-medium">{(data.skill_similarity * 100).toFixed(0)}%</p>
              </div>
              <div className="bg-gray-50 p-3 rounded-lg">
                <p className="text-sm text-gray-500">TF-IDF Similarity</p>
                <p className="text-lg font-medium">{(data.tfidf_similarity * 100).toFixed(0)}%</p>
              </div>
              <div className="bg-gray-50 p-3 rounded-lg">
                <p className="text-sm text-gray-500">Designation</p>
                <p className="text-lg font-medium">{(data.designation_similarity * 100).toFixed(0)}%</p>
              </div>
              <div className="bg-gray-50 p-3 rounded-lg">
                <p className="text-sm text-gray-500">Degree</p>
                <p className="text-lg font-medium">{(data.degree_similarity * 100).toFixed(0)}%</p>
              </div>
            </div>
          </div> */}
        </CardContent>
      </Card>
    </div>
  )
}